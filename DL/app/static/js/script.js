let modelLoaded = false;

// DOM Elements
const trainBtn = document.getElementById('trainBtn');
const loadBtn = document.getElementById('loadBtn');
const predictBtn = document.getElementById('predictBtn');
const modelStatus = document.getElementById('modelStatus');
const titleInput = document.getElementById('title');
const descriptionInput = document.getElementById('description');
const resultSection = document.getElementById('resultSection');
const categoryResult = document.getElementById('categoryResult');
const probabilitiesSection = document.getElementById('probabilitiesSection');
const loading = document.getElementById('loading');
const message = document.getElementById('message');

// Event Listeners
trainBtn.addEventListener('click', trainModel);
loadBtn.addEventListener('click', loadModel);
predictBtn.addEventListener('click', predict);

async function trainModel() {
    showLoading(true);
    hideMessage();
    
    try {
        const response = await fetch('/train', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            }
        });
        
        const data = await response.json();
        
        if (data.success) {
            modelLoaded = true;
            updateModelStatus(true);
            showMessage(`Model trained successfully! Accuracy: ${(data.accuracy * 100).toFixed(2)}%`, 'success');
            console.log('Classification Report:', data.classification_report);
        } else {
            showMessage(`Error: ${data.error}`, 'error');
        }
    } catch (error) {
        showMessage(`Error: ${error.message}`, 'error');
    }
    
    showLoading(false);
}

async function loadModel() {
    showLoading(true);
    hideMessage();
    
    try {
        const response = await fetch('/load_model', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            }
        });
        
        const data = await response.json();
        
        if (data.success) {
            modelLoaded = true;
            updateModelStatus(true);
            showMessage('Model loaded successfully!', 'success');
        } else {
            showMessage(`Error: ${data.error}`, 'error');
        }
    } catch (error) {
        showMessage(`Error: ${error.message}`, 'error');
    }
    
    showLoading(false);
}

async function predict() {
    const title = titleInput.value.trim();
    const description = descriptionInput.value.trim();
    
    if (!title && !description) {
        showMessage('Please enter a title or description', 'error');
        return;
    }
    
    if (!modelLoaded) {
        showMessage('Please train or load the model first', 'error');
        return;
    }
    
    showLoading(true);
    hideMessage();
    resultSection.classList.add('hidden');
    
    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                title: title,
                description: description
            })
        });
        
        const data = await response.json();
        
        if (data.category) {
            displayResult(data);
        } else {
            showMessage(`Error: ${data.error}`, 'error');
        }
    } catch (error) {
        showMessage(`Error: ${error.message}`, 'error');
    }
    
    showLoading(false);
}

function displayResult(data) {
    categoryResult.textContent = data.category;
    resultSection.classList.remove('hidden');
    
    if (data.probabilities) {
        probabilitiesSection.classList.remove('hidden');
        updateProbabilities(data.probabilities);
    } else {
        probabilitiesSection.classList.add('hidden');
    }
}

function updateProbabilities(probs) {
    const categories = ['World', 'Sports', 'Business', 'Sci/Tech'];
    
    categories.forEach(category => {
        const prob = probs[category] || 0;
        const percentage = (prob * 100).toFixed(1);
        
        let elementId;
        if (category === 'Sci/Tech') {
            elementId = 'probSciTech';
        } else {
            elementId = `prob${category}`;
        }
        
        const progressBar = document.getElementById(elementId);
        const textElement = document.getElementById(`${elementId}Text`);
        
        if (progressBar && textElement) {
            progressBar.style.width = `${percentage}%`;
            textElement.textContent = `${percentage}%`;
        }
    });
}

function updateModelStatus(loaded) {
    if (loaded) {
        modelStatus.textContent = 'Model Status: Ready';
        modelStatus.classList.add('ready');
    } else {
        modelStatus.textContent = 'Model Status: Not Loaded';
        modelStatus.classList.remove('ready');
    }
}

function showLoading(show) {
    if (show) {
        loading.classList.remove('hidden');
        trainBtn.disabled = true;
        loadBtn.disabled = true;
        predictBtn.disabled = true;
    } else {
        loading.classList.add('hidden');
        trainBtn.disabled = false;
        loadBtn.disabled = false;
        predictBtn.disabled = false;
    }
}

function showMessage(text, type) {
    message.textContent = text;
    message.className = `message ${type}`;
    message.classList.remove('hidden');
}

function hideMessage() {
    message.classList.add('hidden');
}
