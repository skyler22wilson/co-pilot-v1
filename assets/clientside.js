if (!window.dash_clientside) { window.dash_clientside = {}; }

function setupTextareaAutoResize() {
    const textarea = document.getElementById('partswise-input');
    if (textarea) {
        // Ensures the event listener is added only once
        textarea.removeEventListener('input', adjustTextareaHeight);
        textarea.addEventListener('input', adjustTextareaHeight);

        adjustTextareaHeight.call(textarea); // Adjust height on initialization if there's initial content
        console.log('Textarea event listener added.');
        return true;
    }
    return false;
}

function adjustTextareaHeight() {
    if (this.value === '') {
        this.style.height = '50px'; 
    } else {
        this.style.height = 'auto'; // Reset the height
        this.style.height = (this.scrollHeight) + 'px'; // Set new height based on scroll height
    }
}

// Attempt to set up the textarea immediately, if not found, retry periodically
if (!setupTextareaAutoResize()) {
    const intervalId = setInterval(() => {
        if (setupTextareaAutoResize()) {
            clearInterval(intervalId);
        }
    }, 500);
}

function setupEventListeners() {
    const textArea = document.getElementById('partswise-input');
    const submitButton = document.getElementById('search-button-partswise');

    if (textArea && submitButton) {
        // Ensure this setup runs only once by removing any existing event listeners first
        textArea.removeEventListener('keydown', handleTextareaKeydown);
        // Add the event listener with a named function for easier management
        textArea.addEventListener('keydown', handleTextareaKeydown);
        console.log('Event listeners added for textarea and button.');
        return true; // Indicate that setup was successful
    }
    console.log('Waiting for elements to be available...');
    return false; // Elements not found, setup not complete
}

function handleTextareaKeydown(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault(); // Prevents the default behavior (newline).
        this.value = ''; // Clears the textarea.
        console.log('Enter pressed, attempting to click the button.');
        const submitButton = document.getElementById('search-button-partswise');
        if (submitButton) {
            submitButton.click(); // Simulates button click.
        }
        // Assuming adjustTextareaHeight is defined elsewhere and correctly handles 'this'
        adjustTextareaHeight.call(this);
    }
}

// Attempt to set up the event listeners immediately
if (!setupEventListeners()) {
    // If setup fails, try again periodically until successful
    const retryInterval = setInterval(() => {
        if (setupEventListeners()) {
            clearInterval(retryInterval); // Stop retrying once setup is complete
        }
    }, 100); // Check every 100 milliseconds
}


