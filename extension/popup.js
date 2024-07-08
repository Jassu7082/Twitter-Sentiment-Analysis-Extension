document.getElementById('sentiment-form').addEventListener('submit', function(e) {
  e.preventDefault();
  const text = document.getElementById('tweet-text').value;

  fetch('http://127.0.0.1:5000/predict', {
      method: 'POST',
      headers: {
          'Content-Type': 'application/json'
      },
      body: JSON.stringify({ tweet: text })
  })
  .then(response => response.text())
  .then(data => {
      const resultDiv = document.getElementById('result');
      resultDiv.textContent = 'Sentiment: ' + data;
  })
  .catch(error => console.error('Error:', error));
});
