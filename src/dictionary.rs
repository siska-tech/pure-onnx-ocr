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
        }
    }
}

impl std::error::Error for DictionaryError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            DictionaryError::Io { source, .. } => Some(source),
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
    /// Loads a dictionary from a UTF-8 encoded text file.
    ///
    /// Each non-empty line is treated as a token. Lines containing only
    /// whitespace are ignored. Duplicate tokens result in an error.
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self, DictionaryError> {
        let path = path.as_ref();
        let contents = fs::read_to_string(path).map_err(|source| DictionaryError::Io {
            source,
            path: path.to_path_buf(),
        })?;
        Self::from_str(&contents, Some(path))
    }

    /// Loads a dictionary from a UTF-8 encoded byte slice.
    ///
    /// Each non-empty line is treated as a token. Lines containing only
    /// whitespace are ignored. Duplicate tokens result in an error.
    ///
    /// This method is useful for WASM environments where file system access
    /// is not available.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, DictionaryError> {
        let contents = std::str::from_utf8(bytes).map_err(|_| DictionaryError::Io {
            source: io::Error::new(
                io::ErrorKind::InvalidData,
                "dictionary bytes are not valid UTF-8",
            ),
            path: PathBuf::from("<bytes>"),
        })?;
        Self::from_str(contents, None)
    }

    fn from_str(contents: &str, path: Option<&Path>) -> Result<Self, DictionaryError> {
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
                    path: path
                        .map(|p| p.to_path_buf())
                        .unwrap_or_else(|| PathBuf::from("<bytes>")),
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
                path: path
                    .map(|p| p.to_path_buf())
                    .unwrap_or_else(|| PathBuf::from("<bytes>")),
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

    #[test]
    fn from_bytes_loads_dictionary_successfully() {
        let bytes = b"a\n\nb\n c \n# comment-like text should still be taken literally\n";
        let dictionary = RecDictionary::from_bytes(bytes).unwrap();

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
    }

    #[test]
    fn from_bytes_and_from_path_produce_same_result() {
        let path = unique_temp_file("dict_compare");
        let content = "first\nsecond\nthird\n";
        fs::write(&path, content).unwrap();

        let dict_from_path = RecDictionary::from_path(&path).unwrap();
        let dict_from_bytes = RecDictionary::from_bytes(content.as_bytes()).unwrap();

        assert_eq!(dict_from_path.len(), dict_from_bytes.len());
        assert_eq!(dict_from_path.blank_id(), dict_from_bytes.blank_id());
        for i in 0..dict_from_path.len() {
            assert_eq!(dict_from_path.token(i), dict_from_bytes.token(i));
        }

        fs::remove_file(path).ok();
    }
}
