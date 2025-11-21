// Example: WASMモジュールに埋め込まれたモデルを使用する場合
// このファイルは参考用です。実際にはモデルをビルド時に埋め込む必要があります。

// Import the WASM module
import init, { WasmOcrEngineBuilder, create_engine_with_embedded_models } from './pkg/pure_onnx_ocr.js';

let engine = null;
let wasmInitialized = false;

// DOM elements
const fileInput = document.getElementById('fileInput');
const uploadSection = document.getElementById('uploadSection');
const previewSection = document.getElementById('previewSection');
const previewImage = document.getElementById('previewImage');
const ocrBtn = document.getElementById('ocrBtn');
const resultSection = document.getElementById('resultSection');
const resultTextarea = document.getElementById('result');
const statusMessage = document.getElementById('statusMessage');

// Initialize WASM
async function initializeWasm() {
    if (wasmInitialized) return;

    try {
        showStatus('WASMモジュールを読み込んでいます...', 'loading');
        await init();
        wasmInitialized = true;
        
        // 注意: 実際にモデルを埋め込む場合は、Rustコードで include_bytes! を使用して
        // ビルド時にモデルを埋め込む必要があります。
        // この例では、fetch を使ってモデルファイルを読み込みます。
        showStatus('モデルファイルを読み込んでいます...', 'loading');
        
        // GitHub Pages対応: 現在のページのパスを基準にモデルファイルのパスを構築
        const currentPath = window.location.pathname;
        const basePath = currentPath.substring(0, currentPath.lastIndexOf('/'));
        const modelBasePath = `${basePath}/ppocrv5`;
        
        const [detBytes, recBytes, dictBytes] = await Promise.all([
            fetch(`${modelBasePath}/det.onnx`).then(r => r.arrayBuffer()),
            fetch(`${modelBasePath}/rec.onnx`).then(r => r.arrayBuffer()),
            fetch(`${modelBasePath}/ppocrv5_dict.txt`).then(r => r.arrayBuffer()),
        ]);

        showStatus('OCRエンジンを初期化しています...', 'loading');
        engine = create_engine_with_embedded_models(
            new Uint8Array(detBytes),
            new Uint8Array(recBytes),
            new Uint8Array(dictBytes)
        );

        showStatus('準備完了！画像を選択してください。', 'success');
    } catch (error) {
        showStatus(`初期化に失敗しました: ${error.message}`, 'error');
        console.error('Initialization error:', error);
    }
}

// Read file as ArrayBuffer
function readFileAsArrayBuffer(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = (e) => resolve(e.target.result);
        reader.onerror = (e) => reject(new Error('ファイルの読み込みに失敗しました'));
        reader.readAsArrayBuffer(file);
    });
}

// Handle image file input
fileInput.addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (file) {
        await handleImageFile(file);
    }
});

// Handle drag and drop
uploadSection.addEventListener('dragover', (e) => {
    e.preventDefault();
    e.stopPropagation();
    uploadSection.classList.add('dragover');
});

uploadSection.addEventListener('dragleave', (e) => {
    e.preventDefault();
    e.stopPropagation();
    uploadSection.classList.remove('dragover');
});

uploadSection.addEventListener('drop', async (e) => {
    e.preventDefault();
    e.stopPropagation();
    uploadSection.classList.remove('dragover');

    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
        await handleImageFile(file);
    } else {
        showStatus('画像ファイルを選択してください。', 'error');
    }
});

// Handle image file
async function handleImageFile(file) {
    try {
        currentImageBytes = await readFileAsArrayBuffer(file);
        
        // Show preview
        if (previewSection && previewImage) {
            const blob = new Blob([currentImageBytes], { type: file.type });
            const url = URL.createObjectURL(blob);
            previewImage.src = url;
            previewSection.classList.remove('hidden');
        }
        
        showStatus(`画像を読み込みました: ${file.name}`, 'success');
        updateButtonStates();
    } catch (error) {
        showStatus(`画像の読み込みに失敗しました: ${error.message}`, 'error');
        console.error('Image loading error:', error);
    }
}

// Run OCR
ocrBtn.addEventListener('click', async () => {
    if (!engine) {
        showStatus('エンジンが初期化されていません。', 'error');
        return;
    }

    if (!currentImageBytes) {
        showStatus('画像を選択してください。', 'error');
        return;
    }

    try {
        showStatus('OCR処理を実行しています...', 'loading');
        ocrBtn.disabled = true;

        const imageArray = new Uint8Array(currentImageBytes);
        const resultJson = engine.run_from_bytes(imageArray);
        
        const results = JSON.parse(resultJson);
        displayResults(results);
        
        showStatus('OCR処理が完了しました。', 'success');
    } catch (error) {
        showStatus(`OCR処理に失敗しました: ${error.message}`, 'error');
        console.error('OCR error:', error);
    } finally {
        ocrBtn.disabled = false;
    }
});

// Display results
function displayResults(results) {
    if (!results || results.length === 0) {
        resultTextarea.value = 'テキストが検出されませんでした。';
        resultSection.classList.remove('hidden');
        return;
    }

    let output = '';
    results.forEach((result, index) => {
        output += `[${index + 1}] テキスト: ${result.text}\n`;
        output += `    信頼度: ${(result.confidence * 100).toFixed(2)}%\n`;
        output += `    座標: ${result.bounding_box}\n\n`;
    });

    resultTextarea.value = output;
    resultSection.classList.remove('hidden');
}

function updateButtonStates() {
    ocrBtn.disabled = !wasmInitialized || !engine || !currentImageBytes;
}

function showStatus(message, type = 'info') {
    statusMessage.innerHTML = `<div class="${type}">${message}</div>`;
    
    if (type === 'success' || type === 'error') {
        setTimeout(() => {
            statusMessage.innerHTML = '';
        }, 5000);
    }
}

let currentImageBytes = null;

// Initialize on load
window.addEventListener('load', () => {
    initializeWasm();
});

