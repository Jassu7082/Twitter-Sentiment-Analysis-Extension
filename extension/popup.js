document.getElementById('sentiment-form').addEventListener('submit', function(e) {
    e.preventDefault();
    const text = document.getElementById('tweet-text').value;
  
    fetch('https://localhost:5000/analyze', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ text })
    })
    .then(response => response.json())
    .then(data => {
      const resultDiv = document.getElementById('result');
      resultDiv.textContent = 'Sentiment: ' + data.sentiment;
    })
    .catch(error => console.error('Error:', error));
  });
  