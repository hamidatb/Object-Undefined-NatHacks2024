function playSong(uri) {
    fetch('/play_song', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ uri: uri }),
    })
    .then(response => {
        if (response.status === 200) {
            alert('Playing song!');
        } else {
            alert('Error playing song. Please ensure you are logged into Spotify.');
        }
    })
    .catch((error) => {
        console.error('Error:', error);
    });
}

// Eye-tracking visualization (Placeholder)
window.onload = function() {
    const video = document.createElement('video');
    video.id = 'webcam';
    video.autoplay = true;
    video.style.display = 'none';
    document.body.appendChild(video);

    if (navigator.mediaDevices.getUserMedia) {
        navigator.mediaDevices.getUserMedia({ video: true })
            .then(function(stream) {
                video.srcObject = stream;
                video.play();
            })
            .catch(function(err0r) {
                console.log("Something went wrong with accessing the webcam!");
            });
    }
};


document.addEventListener('DOMContentLoaded', () => {
    var socket = io();

    socket.on('connect', () => {
        console.log('Connected to server');
    });

    socket.on('gaze', data => {
        // data.x and data.y are normalized [0,1]
        // Get window size
        var windowWidth = window.innerWidth;
        var windowHeight = window.innerHeight;

        // Calculate actual position
        var posX = data.x * windowWidth;
        var posY = data.y * windowHeight;

        // Update cursor-dot position
        var cursorDot = document.getElementById('cursor-dot');
        cursorDot.style.transform = `translate(${posX - 10}px, ${posY - 10}px)`; // Adjust for dot size
    });

    socket.on('disconnect', () => {
        console.log('Disconnected from server');
    });
});