use crate::paddle_config::{PaddleConfigError, PaddleInferenceConfig};
use std::{
    collections::HashMap,
    fs, io,
    path::{Path, PathBuf},
};

const BLANK_TOKEN: &str = "blank";

/// Errors that can occur while loading or using the recognition dictionary.
#[derive(Debug)]
pub enum DictionaryError {
    /// The provided path could not be read.
    Io { source: io::Error, path: PathBuf },
    /// The dictionary file did not contain any entries.
    EmptyDictionary { path: PathBuf },
    /// The dictionary file contained a duplicate token.
    DuplicateEntry {
        path: PathBuf,
        line_number: usize,
        token: String,
    },
    /// A PaddleOCR `inference.yml` could not be parsed.
    Config {
        source: PaddleConfigError,
        path: PathBuf,
    },
    /// A PaddleOCR `inference.yml` did not contain `PostProcess.character_dict`.
    MissingCharacterDict { path: PathBuf },
}

impl std::fmt::Display for DictionaryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DictionaryError::Io { source, path } => {
                write!(f, "failed to read dictionary file {:?}: {}", path, source)
            }
            DictionaryError::EmptyDictionary { path } => {
                write!(
                    f,
                    "dictionary file {:?} does not contain any valid entries",
                    path
                )
            }
            DictionaryError::DuplicateEntry {
                path,
                line_number,
                token,
            } => {
                write!(
                    f,
                    "dictionary file {:?} contains duplicate entry {:?} on line {}",
                    path, token, line_number
                )
            }
            DictionaryError::Config { source, path } => {
                write!(f, "failed to read dictionary from {:?}: {}", path, source)
            }
            DictionaryError::MissingCharacterDict { path } => write!(
                f,
                "inference config {:?} does not define PostProcess.character_dict",
                path
            ),
        }
    }
}

impl std::error::Error for DictionaryError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            DictionaryError::Io { source, .. } => Some(source),
            DictionaryError::Config { source, .. } => Some(source),
            _ => None,
        }
    }
}

/// Recognition dictionary wrapping the PaddleOCR vocabulary.
#[derive(Debug, Clone)]
pub struct RecDictionary {
    tokens: Vec<String>,
    reverse: HashMap<String, usize>,
}

impl RecDictionary {
    /// Loads a dictionary from disk.
    ///
    /// Files ending in `.yml` / `.yaml` are treated as PaddleOCR
    /// `inference.yml` configs (see [`RecDictionary::from_inference_yml`]);
    /// everything else is read as a plain text dictionary
    /// (see [`RecDictionary::from_text_file`]).
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self, DictionaryError> {
        let path = path.as_ref();
        if is_yaml_path(path) {
            Self::from_inference_yml(path)
        } else {
            Self::from_text_file(path)
        }
    }

    /// Loads the `PostProcess.character_dict` list from a PaddleOCR
    /// `inference.yml`. PaddleOCR 3.x exports (PP-OCRv5 / PP-OCRv6) embed the
    /// dictionary in this file instead of shipping a separate text file.
    ///
    /// Duplicate characters are accepted to mirror PaddleOCR, which indexes
    /// the list positionally; [`RecDictionary::index_of`] returns the first
    /// occurrence.
    pub fn from_inference_yml(path: impl AsRef<Path>) -> Result<Self, DictionaryError> {
        let path = path.as_ref();
        let config =
            PaddleInferenceConfig::from_path(path).map_err(|source| DictionaryError::Config {
                source,
                path: path.to_path_buf(),
            })?;
        let tokens =
            config
                .character_dict
                .ok_or_else(|| DictionaryError::MissingCharacterDict {
                    path: path.to_path_buf(),
                })?;
        Self::from_tokens(tokens).ok_or_else(|| DictionaryError::EmptyDictionary {
            path: path.to_path_buf(),
        })
    }

    /// Builds a dictionary from an ordered list of characters. Index `0` is
    /// reserved for the CTC blank token and the characters follow from `1`.
    ///
    /// Returns `None` when `tokens` is empty.
    pub fn from_tokens<I, S>(tokens: I) -> Option<Self>
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let blank = BLANK_TOKEN.to_string();
        let mut list = vec![blank.clone()];
        let mut reverse = HashMap::new();
        reverse.insert(blank, 0);
        for token in tokens {
            let token = token.into();
            reverse.entry(token.clone()).or_insert(list.len());
            list.push(token);
        }
        if list.len() == 1 {
            return None;
        }
        Some(Self {
            tokens: list,
            reverse,
        })
    }

    /// Appends the space character as the last class, matching PaddleOCR's
    /// `use_space_char=True` default for `CTCLabelDecode`.
    ///
    /// PaddleOCR recognition models predict `dictionary + 2` classes:
    /// `blank`, every dictionary entry, then `" "`. Without this the last
    /// class decodes as the fallback token (`[UNK]`) instead of a space.
    pub fn with_space_char(mut self) -> Self {
        let index = self.tokens.len();
        self.reverse.entry(" ".to_string()).or_insert(index);
        self.tokens.push(" ".to_string());
        self
    }

    /// Loads a dictionary from a UTF-8 encoded text file.
    ///
    /// Each non-empty line is treated as a token. Lines containing only
    /// whitespace are ignored. Duplicate tokens result in an error.
    pub fn from_text_file(path: impl AsRef<Path>) -> Result<Self, DictionaryError> {
        let path = path.as_ref();
        let contents = fs::read_to_string(path).map_err(|source| DictionaryError::Io {
            source,
            path: path.to_path_buf(),
        })?;

        let mut tokens = Vec::new();
        let mut reverse = HashMap::new();

        let blank = BLANK_TOKEN.to_string();
        reverse.insert(blank.clone(), 0);
        tokens.push(blank);

        for (line_number, raw_line) in contents.lines().enumerate() {
            let line = if line_number == 0 {
                raw_line.trim_start_matches('\u{FEFF}')
            } else {
                raw_line
            };

            let trimmed = line.trim();
            let token = if trimmed.is_empty() && !line.is_empty() {
                line
            } else {
                trimmed
            };

            if token.is_empty() {
                continue;
            }

            if reverse.contains_key(token) {
                return Err(DictionaryError::DuplicateEntry {
                    path: path.to_path_buf(),
                    line_number: line_number + 1,
                    token: token.to_string(),
                });
            }

            let token_string = token.to_string();
            let index = tokens.len();
            reverse.insert(token_string.clone(), index);
            tokens.push(token_string);
        }

        if tokens.len() == 1 {
            return Err(DictionaryError::EmptyDictionary {
                path: path.to_path_buf(),
            });
        }

        Ok(Self { tokens, reverse })
    }

    /// Returns the number of entries in the dictionary.
    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    /// Returns the blank token identifier (always 0).
    pub fn blank_id(&self) -> usize {
        0
    }

    /// Returns the blank token string ("blank").
    pub fn blank_token(&self) -> &str {
        &self.tokens[0]
    }

    /// Returns true if the dictionary has no entries.
    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    /// Returns the token for the given index.
    pub fn token(&self, index: usize) -> Option<&str> {
        self.tokens.get(index).map(|value| value.as_str())
    }

    /// Returns the index for a given token, if it exists.
    pub fn index_of(&self, token: &str) -> Option<usize> {
        self.reverse.get(token).copied()
    }
}

fn is_yaml_path(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| ext.eq_ignore_ascii_case("yml") || ext.eq_ignore_ascii_case("yaml"))
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn unique_temp_file(prefix: &str) -> PathBuf {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("{}_{}.txt", prefix, timestamp))
    }

    #[test]
    fn load_dictionary_successfully() {
        let path = unique_temp_file("dict_success");
        fs::write(
            &path,
            "a\n\nb\n c \n# comment-like text should still be taken literally\n",
        )
        .unwrap();

        let dictionary = RecDictionary::from_path(&path).unwrap();

        assert_eq!(dictionary.len(), 5);
        assert_eq!(dictionary.blank_id(), 0);
        assert_eq!(dictionary.blank_token(), "blank");
        assert_eq!(dictionary.token(0), Some("blank"));
        assert_eq!(dictionary.token(1), Some("a"));
        assert_eq!(dictionary.token(2), Some("b"));
        assert_eq!(dictionary.token(3), Some("c"));
        assert_eq!(
            dictionary.token(4),
            Some("# comment-like text should still be taken literally")
        );
        assert_eq!(dictionary.index_of("blank"), Some(0));
        assert_eq!(dictionary.index_of("c"), Some(3));
        assert!(dictionary.index_of("missing").is_none());

        fs::remove_file(path).ok();
    }

    #[test]
    fn error_on_empty_dictionary() {
        let path = unique_temp_file("dict_empty");
        fs::write(&path, "\n\n\n").unwrap();

        let error = RecDictionary::from_path(&path).unwrap_err();
        match error {
            DictionaryError::EmptyDictionary { .. } => {}
            _ => panic!("expected EmptyDictionary error, got {:?}", error),
        }

        fs::remove_file(path).ok();
    }

    #[test]
    fn error_on_duplicate_entry() {
        let path = unique_temp_file("dict_dup");
        fs::write(&path, "foo\nbar\nfoo\n").unwrap();

        let error = RecDictionary::from_path(&path).unwrap_err();
        match error {
            DictionaryError::DuplicateEntry { line_number, .. } => {
                assert_eq!(line_number, 3);
            }
            _ => panic!("expected DuplicateEntry error, got {:?}", error),
        }

        fs::remove_file(path).ok();
    }

    #[test]
    fn preserves_leading_space_token() {
        let path = unique_temp_file("dict_space");
        // First line is ASCII space, second is fullwidth space, third regular token.
        fs::write(&path, " \n　\nalpha\n").unwrap();

        let dictionary = RecDictionary::from_path(&path).unwrap();

        assert_eq!(dictionary.len(), 4);
        assert_eq!(dictionary.token(0), Some("blank"));
        assert_eq!(dictionary.token(1), Some(" "));
        assert_eq!(dictionary.token(2), Some("　"));
        assert_eq!(dictionary.token(3), Some("alpha"));

        fs::remove_file(path).ok();
    }

    #[test]
    fn dictionary_starts_with_blank_token() {
        let path = unique_temp_file("dict_blank");
        fs::write(&path, "first\nsecond\n").unwrap();

        let dictionary = RecDictionary::from_path(&path).unwrap();

        assert_eq!(dictionary.blank_id(), 0);
        assert_eq!(dictionary.token(0), Some("blank"));
        assert_eq!(dictionary.token(1), Some("first"));
        assert_eq!(dictionary.token(2), Some("second"));

        fs::remove_file(path).ok();
    }
}

#[cfg(test)]
mod paddle_yaml_tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_yaml(contents: &str) -> PathBuf {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("dict_yaml_{}.yml", timestamp));
        fs::write(&path, contents).unwrap();
        path
    }

    #[test]
    fn loads_character_dict_from_inference_yml() {
        let path = temp_yaml(
            "Global:\n  model_name: PP-OCRv6_tiny_rec\nPostProcess:\n  name: CTCLabelDecode\n  character_dict:\n  - '!'\n  - a\n  - 日\n",
        );
        let dictionary = RecDictionary::from_path(&path).unwrap();
        fs::remove_file(&path).ok();

        assert_eq!(dictionary.len(), 4);
        assert_eq!(dictionary.token(0), Some("blank"));
        assert_eq!(dictionary.token(1), Some("!"));
        assert_eq!(dictionary.token(3), Some("日"));
    }

    #[test]
    fn missing_character_dict_is_reported() {
        let path = temp_yaml("PostProcess:\n  name: DBPostProcess\n  thresh: 0.3\n");
        let error = RecDictionary::from_path(&path).unwrap_err();
        fs::remove_file(&path).ok();
        assert!(matches!(
            error,
            DictionaryError::MissingCharacterDict { .. }
        ));
    }

    #[test]
    fn space_char_is_appended_as_last_class() {
        let dictionary = RecDictionary::from_tokens(["a", "b"])
            .unwrap()
            .with_space_char();
        assert_eq!(dictionary.len(), 4);
        assert_eq!(dictionary.token(3), Some(" "));
        assert_eq!(dictionary.index_of(" "), Some(3));
    }

    #[test]
    fn duplicate_tokens_keep_positional_indices() {
        let dictionary = RecDictionary::from_tokens(["a", "b", "a"]).unwrap();
        assert_eq!(dictionary.len(), 4);
        assert_eq!(dictionary.token(3), Some("a"));
        assert_eq!(dictionary.index_of("a"), Some(1));
    }

    #[test]
    fn empty_token_list_is_rejected() {
        assert!(RecDictionary::from_tokens(Vec::<String>::new()).is_none());
    }
}
