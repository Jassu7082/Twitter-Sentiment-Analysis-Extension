document.addEventListener('DOMContentLoaded', () => {
    const observer = new MutationObserver((mutations) => {
      mutations.forEach((mutation) => {
        if (mutation.addedNodes.length) {
          mutation.addedNodes.forEach((node) => {
            if (node.nodeType === 1) {
              analyzeTweets(node);
            }
          });
        }
      });
    });
  
    observer.observe(document.body, { childList: true, subtree: true });
  
    function analyzeTweets(node) {
      const tweets = node.querySelectorAll('article div[lang]');
      tweets.forEach((tweet) => {
        const text = tweet.innerText;
        fetch('http://127.0.0.1:5000/predict', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify({ text })
        })
        .then(response => response.json())
        .then(data => {
          const sentimentLabel = document.createElement('span');
          sentimentLabel.textContent = data.sentiment === 'positive' ? '+ve' : (data.sentiment === 'negative' ? '-ve' : 'neutral');
          sentimentLabel.style.position = 'absolute';
          sentimentLabel.style.top = '0';
          sentimentLabel.style.right = '0';
          sentimentLabel.style.backgroundColor = 'yellow';
          sentimentLabel.style.padding = '2px';
          sentimentLabel.style.fontSize = '10px';
          tweet.appendChild(sentimentLabel);
        })
        .catch(error => console.error('Error:', error));
      });
    }
  
    analyzeTweets(document.body);
  });
  