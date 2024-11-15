document.addEventListener('DOMContentLoaded', () => {
    const socket = io(); // Ensure this works with the updated Socket.IO CDN

    // DOM Elements
    const cursorDot = document.getElementById('cursor-dot');
    const quadrantInfo = document.getElementById('quadrant-info');
    const buttons = {
        'top-left': document.getElementById('button-top-left'),
        'top-right': document.getElementById('button-top-right'),
        'bottom-left': document.getElementById('button-bottom-left'),
        'bottom-right': document.getElementById('button-bottom-right'),
    };

    // Variables to track gaze duration
    let currentQuadrant = null;
    let gazeStartTime = null;
    const GAZE_HOLD_TIME = 5000; // 5 seconds in milliseconds

    // Function to move cursor to a specific corner
    function moveCursorToCorner(corner) {
        if (!cursorDot) return;

        const viewportWidth = window.innerWidth;
        const viewportHeight = window.innerHeight;

        let x = 0, y = 0; // Default to top-left

        switch (corner) {
            case 'top-left':
                x = 0;
                y = 0;
                break;
            case 'top-right':
                x = viewportWidth - cursorDot.offsetWidth;
                y = 0;
                break;
            case 'bottom-left':
                x = 0;
                y = viewportHeight - cursorDot.offsetHeight;
                break;
            case 'bottom-right':
                x = viewportWidth - cursorDot.offsetWidth;
                y = viewportHeight - cursorDot.offsetHeight;
                break;
            default:
                console.warn('Invalid corner specified');
                return;
        }

        cursorDot.style.transform = `translate(${x}px, ${y}px)`;
        //console.log(`Moved cursor to ${corner}`);
    }

    // Function to determine which quadrant the gaze is in
    function getQuadrant(x, y) {
        const viewportWidth = window.innerWidth;
        const viewportHeight = window.innerHeight;

        const isLeft = x < viewportWidth / 2;
        const isTop = y < viewportHeight / 2;

        if (isTop && isLeft) {
            return 'top-left';
        } else if (isTop && !isLeft) {
            return 'top-right';
        } else if (!isTop && isLeft) {
            return 'bottom-left';
        } else {
            return 'bottom-right';
        }
    }


    // Handle gaze data received from the server
    socket.on('gaze', data => {
       //  console.log(`Received gaze data: Quadrant=${data.quadrant}, x=${data.x}, y=${data.y}`);

        // Determine which quadrant the gaze data falls into
        const quadrant = getQuadrant(data.x, data.y);
        
        // Move cursor to the corresponding corner
        moveCursorToCorner(quadrant);

        // Update quadrant information display
        if (quadrantInfo) {
            quadrantInfo.textContent = `Quadrant: ${quadrant}`;
        }

        // Gaze detection logic
        if (quadrant !== currentQuadrant) {
            // Quadrant changed, reset timer
            currentQuadrant = quadrant;
            gazeStartTime = Date.now();
        } else {
            // Same quadrant, check if gaze duration meets threshold
            const elapsedTime = Date.now() - gazeStartTime;
            if (elapsedTime >= GAZE_HOLD_TIME) {
                // Trigger button click
                triggerButtonClick(quadrant);
                // Reset timer to prevent multiple triggers
                gazeStartTime = Date.now();
            }
        }
    });

    function triggerButtonClick(quadrant) {
        const button = buttons[quadrant];
        if (button) {
            console.log(`Triggering click on ${quadrant} button`);
            button.click(); // Simulate button click
            // Optionally, provide visual feedback
            button.classList.add('active');
            setTimeout(() => {
                button.classList.remove('active');
            }, 200);
        } else {
            console.warn(`No button found for quadrant ${quadrant}`);
        }
    }
    /// For debugging: Add red border to visualize viewport bounds
    /* const createBorderElement = (top, left, width, height) => {
        const borderElement = document.createElement('div');
        borderElement.style.position = 'fixed';
        borderElement.style.top = top;
        borderElement.style.left = left;
        borderElement.style.width = width;
        borderElement.style.height = height;
        borderElement.style.backgroundColor = 'red'; // Use solid red color
        borderElement.style.pointerEvents = 'none';
        borderElement.style.zIndex = '900'; // Below the cursor dot, above the main content
        document.body.appendChild(borderElement);
    };

    // Create borders for all four sides
    createBorderElement('0', '0', '100%', '4px'); // Top border
    createBorderElement('0', '0', '4px', '100%'); // Left border
    createBorderElement('0', 'calc(100% - 4px)', '4px', '100%'); // Right border
    createBorderElement('calc(100% - 4px)', '0', '100%', '4px'); // Bottom border
    */ 

    // Handle connection events
    socket.on('connect', () => {
        console.log('Connected to server');
    });

    socket.on('disconnect', () => {
        console.log('Disconnected from server');
    });
});
