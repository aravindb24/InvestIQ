const API_KEY = '4NIFR0WSYEG706YI';

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  const { ticker } = message;
  const apiUrl = `https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol=${ticker}&apikey=${API_KEY}`;

  fetch(apiUrl)
    .then(response => response.json())
    .then(data => {
      if (data['Global Quote'] && data['Global Quote']['05. price']) {
        const price = data['Global Quote']['05. price'];
        sendResponse({ price });
      } else {
        sendResponse({ error: 'Invalid ticker symbol or API limit reached.' });
      }
    })
    .catch(error => {
      sendResponse({ error: error.message });
    });

  // Return true to indicate that the response will be sent asynchronously
  return true;
});
