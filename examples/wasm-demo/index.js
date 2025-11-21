// Import the WASM module
// Note: After running `wasm-pack build --target web --features wasm`, 
// the module will be available at ./pkg/pure_onnx_ocr.js
import init, { WasmOcrEngineBuilder } from './pkg/pure_onnx_ocr.js';

let engine = null;
let wasmInitialized = false;

// DOM elements (will be initialized after DOM is loaded)
let modelSourceSelect;
let modelUploadSection;
let detModelInput;
let recModelInput;
let dictionaryInput;
let fileInput;
let uploadSection;
let previewSection;
let previewImage;
let previewCanvas;
let previewLoadingOverlay;
let ocrBtn;
let clearBtn;
let resultSection;
let resultTextarea;
let copyBtn;
let downloadBtn;
let statusMessage;
let detLimitSideLen;
let detUnclipRatio;
let recBatchSize;

// Initialize DOM elements
function initializeDOMElements() {
    modelSourceSelect = document.getElementById('modelSource');
    modelUploadSection = document.getElementById('modelUploadSection');
    detModelInput = document.getElementById('detModel');
    recModelInput = document.getElementById('recModel');
    dictionaryInput = document.getElementById('dictionary');
    fileInput = document.getElementById('fileInput');
    uploadSection = document.getElementById('uploadSection');
    previewSection = document.getElementById('previewSection');
    previewImage = document.getElementById('previewImage');
    previewCanvas = document.getElementById('previewCanvas');
    previewLoadingOverlay = document.getElementById('previewLoadingOverlay');
    ocrBtn = document.getElementById('ocrBtn');
    clearBtn = document.getElementById('clearBtn');
    resultSection = document.getElementById('resultSection');
    resultTextarea = document.getElementById('result');
    copyBtn = document.getElementById('copyBtn');
    downloadBtn = document.getElementById('downloadBtn');
    statusMessage = document.getElementById('statusMessage');
    detLimitSideLen = document.getElementById('detLimitSideLen');
    detUnclipRatio = document.getElementById('detUnclipRatio');
    recBatchSize = document.getElementById('recBatchSize');
    
    // Verify critical elements exist
    if (!fileInput || !uploadSection || !ocrBtn) {
        console.error('Critical DOM elements not found!');
        return false;
    }
    return true;
}

// Model files
let detModelBytes = null;
let recModelBytes = null;
let dictionaryBytes = null;
let currentImageBytes = null;
let currentOcrResults = null; // Store OCR results for redrawing

// Model source: 'fetch' or 'upload'
let modelSource = 'fetch';

// Initialize WASM
async function initializeWasm() {
    if (wasmInitialized) return;

    try {
        showStatus('WASMモジュールを読み込んでいます...', 'loading');
        await init();
        wasmInitialized = true;
        showStatus('WASMモジュールの読み込みが完了しました。', 'success');
        
        // デフォルトでサーバーからモデルを読み込む
        if (modelSource === 'fetch') {
            await loadModelsFromServer();
        }
        
        updateButtonStates();
    } catch (error) {
        showStatus(`WASMモジュールの読み込みに失敗しました: ${error.message}`, 'error');
        console.error('WASM initialization error:', error);
    }
}

// Load models from server using fetch
async function loadModelsFromServer() {
    try {
        showStatus('モデルファイルをサーバーから読み込んでいます...', 'loading');
        
        // GitHub Pages対応: 現在のページのパスを基準にモデルファイルのパスを構築
        // 例: /pure-onnx-ocr/examples/wasm-demo/ の場合、./ppocrv5 が正しく解決される
        // 現在のHTMLファイルのディレクトリを取得
        const currentPath = window.location.pathname;
        const basePath = currentPath.substring(0, currentPath.lastIndexOf('/'));
        const modelBasePath = `${basePath}/ppocrv5`;
        
        console.log('Loading models from:', modelBasePath);
        
        const [detResponse, recResponse, dictResponse] = await Promise.all([
            fetch(`${modelBasePath}/det.onnx`),
            fetch(`${modelBasePath}/rec.onnx`),
            fetch(`${modelBasePath}/ppocrv5_dict.txt`),
        ]);

        if (!detResponse.ok || !recResponse.ok || !dictResponse.ok) {
            throw new Error('モデルファイルが見つかりません。サーバーにモデルファイルを配置するか、ファイルアップロードを使用してください。');
        }

        detModelBytes = await detResponse.arrayBuffer();
        recModelBytes = await recResponse.arrayBuffer();
        dictionaryBytes = await dictResponse.arrayBuffer();

        showStatus('モデルファイルの読み込みが完了しました。', 'success');
        
        // エンジンを再構築
        await loadModels();
    } catch (error) {
        showStatus(`モデルファイルの読み込みに失敗しました: ${error.message}`, 'error');
        console.error('Model loading error:', error);
        // エラーの場合はアップロードモードに切り替えを促す
        if (modelSourceSelect) {
            modelSourceSelect.value = 'upload';
            onModelSourceChange();
        }
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

// Load model files
async function loadModels() {
    if (!detModelBytes || !recModelBytes || !dictionaryBytes) {
        return false;
    }

    try {
        showStatus('OCRエンジンを初期化しています...', 'loading');
        const builder = new WasmOcrEngineBuilder();

        const detLimit = detLimitSideLen ? (parseInt(detLimitSideLen.value) || 960) : 960;
        const unclipRatio = detUnclipRatio ? (parseFloat(detUnclipRatio.value) || 1.5) : 1.5;
        const batchSize = recBatchSize ? (parseInt(recBatchSize.value) || 8) : 8;

        engine = builder
            .det_model_bytes(new Uint8Array(detModelBytes))
            .rec_model_bytes(new Uint8Array(recModelBytes))
            .dictionary_bytes(new Uint8Array(dictionaryBytes))
            .det_limit_side_len(detLimit)
            .det_unclip_ratio(unclipRatio)
            .rec_batch_size(batchSize)
            .build();

        showStatus('OCRエンジンの初期化が完了しました。', 'success');
        return true;
    } catch (error) {
        showStatus(`OCRエンジンの初期化に失敗しました: ${error.message}`, 'error');
        console.error('Engine initialization error:', error);
        return false;
    }
}

// Handle model source change
function onModelSourceChange() {
    modelSource = modelSourceSelect ? modelSourceSelect.value : 'fetch';
    
    if (modelSource === 'upload') {
        if (modelUploadSection) {
            modelUploadSection.classList.remove('hidden');
        }
        // アップロードモードに切り替えた場合、既にアップロード済みのファイルがある場合は保持する
        // エンジンは再構築が必要なためクリア
        engine = null;
        showStatus('ファイルアップロードモードに切り替えました。各ファイルを個別に選択できます。', 'info');
    } else {
        if (modelUploadSection) {
            modelUploadSection.classList.add('hidden');
        }
        // fetchモードに戻った場合は、サーバーから再度読み込む
        loadModelsFromServer();
    }
    
    updateButtonStates();
}

// Setup event listeners
function setupEventListeners() {
    console.log('Setting up event listeners...');
    console.log('fileInput:', fileInput);
    console.log('uploadSection:', uploadSection);
    console.log('ocrBtn:', ocrBtn);
    
    if (!modelSourceSelect || !fileInput || !uploadSection) {
        console.error('Cannot setup event listeners: DOM elements not found');
        console.error('modelSourceSelect:', modelSourceSelect);
        console.error('fileInput:', fileInput);
        console.error('uploadSection:', uploadSection);
        return;
    }
    
    // Handle model source change
    modelSourceSelect.addEventListener('change', onModelSourceChange);

    // Handle model file inputs - 個別のファイルを切り替え可能にする
    if (detModelInput) {
        detModelInput.addEventListener('change', async (e) => {
            const file = e.target.files[0];
            if (file) {
                try {
                    detModelBytes = await readFileAsArrayBuffer(file);
                    showStatus('検出モデルを読み込みました。', 'success');
                    // 他のファイルも読み込まれている場合はエンジンを再構築
                    if (recModelBytes && dictionaryBytes) {
                        await loadModels();
                    } else {
                        showStatus('認識モデルと辞書ファイルも読み込む必要があります。', 'info');
                    }
                    updateButtonStates();
                } catch (error) {
                    showStatus(`検出モデルの読み込みに失敗しました: ${error.message}`, 'error');
                }
            }
        });
    }
    
    if (recModelInput) {
        recModelInput.addEventListener('change', async (e) => {
            const file = e.target.files[0];
            if (file) {
                try {
                    recModelBytes = await readFileAsArrayBuffer(file);
                    showStatus('認識モデルを読み込みました。', 'success');
                    // 他のファイルも読み込まれている場合はエンジンを再構築
                    if (detModelBytes && dictionaryBytes) {
                        await loadModels();
                    } else {
                        showStatus('検出モデルと辞書ファイルも読み込む必要があります。', 'info');
                    }
                    updateButtonStates();
                } catch (error) {
                    showStatus(`認識モデルの読み込みに失敗しました: ${error.message}`, 'error');
                }
            }
        });
    }
    
    if (dictionaryInput) {
        dictionaryInput.addEventListener('change', async (e) => {
            const file = e.target.files[0];
            if (file) {
                try {
                    dictionaryBytes = await readFileAsArrayBuffer(file);
                    showStatus('辞書ファイルを読み込みました。', 'success');
                    // 他のファイルも読み込まれている場合はエンジンを再構築
                    if (detModelBytes && recModelBytes) {
                        await loadModels();
                    } else {
                        showStatus('検出モデルと認識モデルも読み込む必要があります。', 'info');
                    }
                    updateButtonStates();
                } catch (error) {
                    showStatus(`辞書ファイルの読み込みに失敗しました: ${error.message}`, 'error');
                }
            }
        });
    }
    
    // Handle image file input
    fileInput.addEventListener('change', async (e) => {
        console.log('File input change event fired', e.target.files);
        const file = e.target.files[0];
        if (file) {
            console.log('File selected:', file.name, file.type);
            await handleImageFile(file);
        } else {
            console.warn('No file selected');
        }
    });
    
    // Prevent fileInput from interfering with drag and drop
    fileInput.addEventListener('dragover', (e) => {
        e.preventDefault();
        e.stopPropagation();
    });
    
    fileInput.addEventListener('drop', (e) => {
        e.preventDefault();
        e.stopPropagation();
        // Let the uploadSection handle the drop
    });
    
    // Handle drag and drop on upload section
    uploadSection.addEventListener('dragover', (e) => {
        console.log('Drag over event fired');
        e.preventDefault();
        e.stopPropagation();
        uploadSection.classList.add('dragover');
    });

    uploadSection.addEventListener('dragleave', (e) => {
        console.log('Drag leave event fired');
        e.preventDefault();
        e.stopPropagation();
        uploadSection.classList.remove('dragover');
    });

    uploadSection.addEventListener('drop', async (e) => {
        console.log('Drop event fired', e.dataTransfer.files);
        e.preventDefault();
        e.stopPropagation();
        uploadSection.classList.remove('dragover');

        const file = e.dataTransfer.files[0];
        if (file) {
            console.log('File dropped:', file.name, file.type);
            if (file.type.startsWith('image/')) {
                await handleImageFile(file);
            } else {
                showStatus('画像ファイルを選択してください。', 'error');
            }
        } else {
            console.warn('No file in drop event');
            showStatus('画像ファイルを選択してください。', 'error');
        }
    });
    
    // Run OCR
    ocrBtn.addEventListener('click', async () => {
        // エンジンがない、またはモデルが変更された場合は再構築
        if (!engine || !detModelBytes || !recModelBytes || !dictionaryBytes) {
            const success = await loadModels();
            if (!success) {
                return;
            }
        }

        if (!currentImageBytes) {
            showStatus('画像を選択してください。', 'error');
            return;
        }

        try {
            showStatus('OCR処理を実行しています...', 'loading');
            ocrBtn.disabled = true;
            
            // Change button text to show processing state
            const originalButtonText = ocrBtn.textContent;
            ocrBtn.textContent = '処理中...';
            ocrBtn.classList.add('processing');
            
            // Show loading overlay on preview
            if (previewLoadingOverlay && previewSection && !previewSection.classList.contains('hidden')) {
                previewLoadingOverlay.classList.remove('hidden');
            }
            
            // Add processing class to body to prevent interactions
            document.body.classList.add('processing');

            // Use setTimeout to allow UI to update before starting heavy computation
            // This gives the browser a chance to render the loading state
            // Note: WASM execution is synchronous and will block the UI thread
            // This is a limitation of the current implementation
            await new Promise(resolve => setTimeout(resolve, 100));
            
            // Force a repaint to show loading state
            if (previewLoadingOverlay) {
                previewLoadingOverlay.offsetHeight; // Force reflow
            }
            
            // Use requestAnimationFrame to ensure UI is updated
            await new Promise(resolve => requestAnimationFrame(resolve));
            
            // Run OCR - this will block the UI thread, but at least the loading state is visible
            const imageArray = new Uint8Array(currentImageBytes);
            const resultJson = engine.run_from_bytes(imageArray);
            
            const results = JSON.parse(resultJson);
            currentOcrResults = results; // Store results for redrawing
            displayResults(results);
            
            showStatus('OCR処理が完了しました。', 'success');
        } catch (error) {
            showStatus(`OCR処理に失敗しました: ${error.message}`, 'error');
            console.error('OCR error:', error);
        } finally {
            // Remove processing class from body
            document.body.classList.remove('processing');
            
            // Restore button state
            ocrBtn.disabled = false;
            ocrBtn.textContent = 'OCR実行';
            ocrBtn.classList.remove('processing');
            
            // Hide loading overlay
            if (previewLoadingOverlay) {
                previewLoadingOverlay.classList.add('hidden');
            }
            
            updateButtonStates();
        }
    });
    
    // Copy results
    if (copyBtn) {
        copyBtn.addEventListener('click', () => {
            if (resultTextarea) {
                resultTextarea.select();
                document.execCommand('copy');
                showStatus('結果をクリップボードにコピーしました。', 'success');
            }
        });
    }
    
    // Download results
    if (downloadBtn) {
        downloadBtn.addEventListener('click', () => {
            if (resultTextarea) {
                const blob = new Blob([resultTextarea.value], { type: 'text/plain' });
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = 'ocr_results.txt';
                a.click();
                URL.revokeObjectURL(url);
                showStatus('結果をダウンロードしました。', 'success');
            }
        });
    }
    
    // Clear
    if (clearBtn) {
        clearBtn.addEventListener('click', () => {
            if (modelSource === 'upload') {
                if (detModelInput) detModelInput.value = '';
                if (recModelInput) recModelInput.value = '';
                if (dictionaryInput) dictionaryInput.value = '';
                detModelBytes = null;
                recModelBytes = null;
                dictionaryBytes = null;
                engine = null;
            }
            
            if (fileInput) fileInput.value = '';
            currentImageBytes = null;
            currentOcrResults = null;
            
            if (previewSection) previewSection.classList.add('hidden');
            if (resultSection) resultSection.classList.add('hidden');
            if (statusMessage) statusMessage.innerHTML = '';
            
            // Clear canvas
            if (previewCanvas) {
                const ctx = previewCanvas.getContext('2d');
                if (ctx) {
                    ctx.clearRect(0, 0, previewCanvas.width, previewCanvas.height);
                }
            }
            
            updateButtonStates();
            
            if (modelSource === 'fetch') {
                showStatus('モデルファイルをクリアしました。', 'info');
            } else {
                showStatus('モデルファイルと画像をクリアしました。', 'info');
            }
        });
    }
}

// Handle image file
async function handleImageFile(file) {
    console.log('handleImageFile called with:', file);
    try {
        if (!file) {
            console.error('No file provided to handleImageFile');
            showStatus('ファイルが選択されていません。', 'error');
            return;
        }
        
        console.log('Reading file as ArrayBuffer...');
        currentImageBytes = await readFileAsArrayBuffer(file);
        console.log('File read successfully, size:', currentImageBytes.byteLength);
        
        // Show preview
        if (previewImage && previewSection) {
            console.log('Creating preview...');
            const blob = new Blob([currentImageBytes], { type: file.type });
            const url = URL.createObjectURL(blob);
            previewImage.src = url;
            previewSection.classList.remove('hidden');
            
            // Setup canvas when image is loaded
            if (previewCanvas) {
                const setupCanvas = () => {
                    // Set canvas size to match image display size
                    const rect = previewImage.getBoundingClientRect();
                    previewCanvas.width = rect.width;
                    previewCanvas.height = rect.height;
                    const ctx = previewCanvas.getContext('2d');
                    if (ctx) {
                        ctx.clearRect(0, 0, previewCanvas.width, previewCanvas.height);
                    }
                };
                
                previewImage.onload = setupCanvas;
                
                // Also update on window resize
                const resizeHandler = () => {
                    if (previewSection && !previewSection.classList.contains('hidden')) {
                        setupCanvas();
                        // Redraw polygons if we have results
                        if (currentOcrResults) {
                            drawPolygonsOnCanvas(currentOcrResults);
                        }
                    }
                };
                window.addEventListener('resize', resizeHandler);
            }
            
            console.log('Preview created');
        } else {
            console.warn('Preview elements not found:', { previewImage, previewSection });
        }
        
        showStatus(`画像を読み込みました: ${file.name}`, 'success');
        updateButtonStates();
        console.log('Image file handled successfully');
    } catch (error) {
        console.error('Image file handling error:', error);
        showStatus(`画像の読み込みに失敗しました: ${error.message}`, 'error');
    }
}

// Display results
function displayResults(results) {
    if (!resultTextarea || !resultSection) return;
    
    if (!results || results.length === 0) {
        resultTextarea.value = 'テキストが検出されませんでした。';
        resultSection.classList.remove('hidden');
        drawPolygonsOnCanvas([]);
        return;
    }

    let output = '';
    results.forEach((result, index) => {
        output += `[${index + 1}] テキスト: ${result.text}\n`;
        output += `    信頼度: ${(result.confidence * 100).toFixed(2)}%\n\n`;
    });

    resultTextarea.value = output;
    resultSection.classList.remove('hidden');
    
    // Draw polygons on canvas
    drawPolygonsOnCanvas(results);
}

// Draw polygons on canvas overlay
function drawPolygonsOnCanvas(results) {
    if (!previewCanvas || !previewImage) return;
    
    const img = previewImage;
    
    // Wait for image to load
    if (!img.complete || img.naturalWidth === 0) {
        img.onload = () => drawPolygonsOnCanvas(results);
        return;
    }
    
    // Set canvas size to match image display size
    const rect = img.getBoundingClientRect();
    previewCanvas.width = rect.width;
    previewCanvas.height = rect.height;
    
    const ctx = previewCanvas.getContext('2d');
    if (!ctx) return;
    
    // Clear canvas
    ctx.clearRect(0, 0, previewCanvas.width, previewCanvas.height);
    
    if (!results || results.length === 0) return;
    
    // Calculate scale factors
    const scaleX = previewCanvas.width / img.naturalWidth;
    const scaleY = previewCanvas.height / img.naturalHeight;
    
    // Draw each polygon
    results.forEach((result, index) => {
        if (!result.bounding_box) {
            console.warn('No bounding_box for result:', result);
            return;
        }
        
        // Ensure bounding_box is an array
        let boundingBox = result.bounding_box;
        if (!Array.isArray(boundingBox)) {
            console.warn('bounding_box is not an array:', boundingBox);
            return;
        }
        
        if (boundingBox.length === 0) return;
        
        // Use different colors for each polygon
        const hue = (index * 137.5) % 360; // Golden angle for color distribution
        ctx.strokeStyle = `hsl(${hue}, 70%, 50%)`;
        ctx.fillStyle = `hsla(${hue}, 70%, 50%, 0.2)`;
        ctx.lineWidth = 2;
        
        // Draw outer ring (first element)
        const outerRing = boundingBox[0];
        if (!Array.isArray(outerRing)) {
            console.warn('outerRing is not an array:', outerRing);
            return;
        }
        
        if (outerRing.length > 0) {
            ctx.beginPath();
            const firstPoint = outerRing[0];
            // Support both array format [x, y] and object format {x, y}
            const firstX = (typeof firstPoint === 'object' && 'x' in firstPoint) ? firstPoint.x : (Array.isArray(firstPoint) ? firstPoint[0] : null);
            const firstY = (typeof firstPoint === 'object' && 'y' in firstPoint) ? firstPoint.y : (Array.isArray(firstPoint) ? firstPoint[1] : null);
            
            if (firstX === null || firstY === null) {
                console.warn('Invalid first point:', firstPoint);
                return;
            }
            
            ctx.moveTo(firstX * scaleX, firstY * scaleY);
            
            for (let i = 1; i < outerRing.length; i++) {
                const point = outerRing[i];
                const x = (typeof point === 'object' && 'x' in point) ? point.x : (Array.isArray(point) ? point[0] : null);
                const y = (typeof point === 'object' && 'y' in point) ? point.y : (Array.isArray(point) ? point[1] : null);
                if (x !== null && y !== null) {
                    ctx.lineTo(x * scaleX, y * scaleY);
                }
            }
            
            ctx.closePath();
            ctx.fill();
            ctx.stroke();
            
            // Draw text label
            if (result.text) {
                let sumX = 0, sumY = 0;
                let validPoints = 0;
                for (const point of outerRing) {
                    const x = (typeof point === 'object' && 'x' in point) ? point.x : (Array.isArray(point) ? point[0] : null);
                    const y = (typeof point === 'object' && 'y' in point) ? point.y : (Array.isArray(point) ? point[1] : null);
                    if (x !== null && y !== null) {
                        sumX += x;
                        sumY += y;
                        validPoints++;
                    }
                }
                
                if (validPoints > 0) {
                    const centerX = (sumX / validPoints) * scaleX;
                    const centerY = (sumY / validPoints) * scaleY;
                    
                    ctx.fillStyle = `hsl(${hue}, 70%, 50%)`;
                    ctx.font = '14px sans-serif';
                    ctx.textAlign = 'center';
                    ctx.textBaseline = 'middle';
                    ctx.fillText(result.text, centerX, centerY);
                }
            }
        }
        
        // Draw inner rings (holes) if any
        for (let i = 1; i < boundingBox.length; i++) {
            const innerRing = boundingBox[i];
            if (!Array.isArray(innerRing) || innerRing.length === 0) continue;
            
            ctx.beginPath();
            const firstPoint = innerRing[0];
            const firstX = (typeof firstPoint === 'object' && 'x' in firstPoint) ? firstPoint.x : (Array.isArray(firstPoint) ? firstPoint[0] : null);
            const firstY = (typeof firstPoint === 'object' && 'y' in firstPoint) ? firstPoint.y : (Array.isArray(firstPoint) ? firstPoint[1] : null);
            
            if (firstX !== null && firstY !== null) {
                ctx.moveTo(firstX * scaleX, firstY * scaleY);
                
                for (let j = 1; j < innerRing.length; j++) {
                    const point = innerRing[j];
                    const x = (typeof point === 'object' && 'x' in point) ? point.x : (Array.isArray(point) ? point[0] : null);
                    const y = (typeof point === 'object' && 'y' in point) ? point.y : (Array.isArray(point) ? point[1] : null);
                    if (x !== null && y !== null) {
                        ctx.lineTo(x * scaleX, y * scaleY);
                    }
                }
                
                ctx.closePath();
                ctx.stroke();
            }
        }
    });
}

// Update button states
function updateButtonStates() {
    if (!ocrBtn) return;
    
    const hasModels = detModelBytes && recModelBytes && dictionaryBytes;
    const canRunOcr = wasmInitialized && hasModels && currentImageBytes;
    ocrBtn.disabled = !canRunOcr;
}

// Show status message
function showStatus(message, type = 'info') {
    if (!statusMessage) {
        console.log(`[${type}] ${message}`);
        return;
    }
    
    statusMessage.innerHTML = `<div class="${type}">${message}</div>`;
    
    if (type === 'success' || type === 'error') {
        setTimeout(() => {
            if (statusMessage) {
                statusMessage.innerHTML = '';
            }
        }, 5000);
    }
}

// Initialize on load
window.addEventListener('load', () => {
    console.log('Window loaded, initializing...');
    
    // Initialize DOM elements first
    if (!initializeDOMElements()) {
        console.error('Failed to initialize DOM elements');
        return;
    }
    
    console.log('DOM elements initialized successfully');
    
    // Setup event listeners
    setupEventListeners();
    
    console.log('Event listeners set up');
    
    // Initialize WASM
    initializeWasm();
});
