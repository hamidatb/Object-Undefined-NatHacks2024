// static/js/script.js

document.addEventListener('DOMContentLoaded', () => {
    const socket = io(); // Ensure this works with the updated Socket.IO CDN

    // DOM Elements
    const cursorDot = document.getElementById('cursor-dot');
    const quadrantInfo = document.getElementById('quadrant-info');

    // Function to update quadrant highlights
    // function updateQuadrantHighlight(quadrant) {
        // Clear all highlights
        // document.body.style.background = '#FFE695';

        // Highlight the quadrant
        //if (quadrant === 'top_left') {
            //document.body.style.background = 'linear-gradient(to bottom right, rgba(0, 255, 0, 0.3), transparent)';
        //} else if (quadrant === 'top_right') {
            //document.body.style.background = 'linear-gradient(to bottom left, rgba(0, 255, 0, 0.3), transparent)';
        //} else if (quadrant === 'bottom_left') {
            //document.body.style.background = 'linear-gradient(to top right, rgba(0, 255, 0, 0.3), transparent)';
        //} else if (quadrant === 'bottom_right') {
            //document.body.style.background = 'linear-gradient(to top left, rgba(0, 255, 0, 0.3), transparent)';
        //}
    //}

    // Handle gaze data received from the server
    socket.on('gaze', data => {
        console.log(`Received gaze data: Quadrant=${data.quadrant}, x=${data.x}, y=${data.y}`);

        // Update cursor-dot position
        if (cursorDot) {
            const viewportWidth = document.documentElement.clientWidth;
            const viewportHeight = document.documentElement.clientHeight;

            const x = Math.min(Math.max(data.x, 0), viewportWidth);
            const y = Math.min(Math.max(data.y, 0), viewportHeight);
            
            cursorDot.style.transform = `translate(${x}px, ${y}px)`;
        }

        // Update quadrant information display
        if (quadrantInfo) {
            quadrantInfo.textContent = `Quadrant: ${data.quadrant}`;
        }

        // Highlight the quadrant on the webpage
        updateQuadrantHighlight(data.quadrant);
    });

    // Handle connection events
    socket.on('connect', () => {
        console.log('Connected to server');
    });

    socket.on('disconnect', () => {
        console.log('Disconnected from server');
    });
});
