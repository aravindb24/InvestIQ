document.getElementById('fetchButton').addEventListener('click', () => {
    const ticker = document.getElementById('tickerInput').value.toUpperCase();
    const resultDiv = document.getElementById('result');
    resultDiv.textContent = 'Fetching...';
  
    // Send message to background script
    chrome.runtime.sendMessage({ ticker }, (response) => {
      if (response.error) {
        resultDiv.textContent = `Error: ${response.error}`;
      } else {
        resultDiv.textContent = `The current price of ${ticker} is $${response.price}`;
      }
    });
  });
  