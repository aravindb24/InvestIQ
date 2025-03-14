document.addEventListener('DOMContentLoaded', () => {
  const apiKey = '4NIFR0WSYEG706YI'; // Replace with your API key
  const searchBtn = document.getElementById('searchBtn');
  const stockSymbolInput = document.getElementById('stockSymbol');
  const resultContainer = document.getElementById('resultContainer');
  const errorContainer = document.getElementById('errorContainer');
  const stockName = document.getElementById('stockName');
  const stockPrice = document.getElementById('stockPrice');
  const stockChange = document.getElementById('stockChange');
  const stockVolume = document.getElementById('stockVolume');
  
  searchBtn.addEventListener('click', fetchStockData);
  stockSymbolInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') fetchStockData();
  });
  
  function fetchStockData() {
    const symbol = stockSymbolInput.value.trim().toUpperCase();
    if (!symbol) {
      showError('Please enter a stock symbol.');
      return;
    }
    
    const url = `https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol=${symbol}&apikey=${apiKey}`;
    
    fetch(url)
      .then(response => response.json())
      .then(data => {
        if (data['Global Quote'] && Object.keys(data['Global Quote']).length > 0) {
          displayStockData(data['Global Quote']);
        } else {
          showError('Invalid stock symbol or no data available.');
        }
      })
      .catch(error => {
        console.error('Error fetching stock data:', error);
        showError('Failed to fetch stock data. Please try again.');
      });
  }
  
  function displayStockData(quoteData) {
    errorContainer.style.display = 'none';
    resultContainer.style.display = 'block';
    
    stockName.textContent = quoteData['01. symbol'];
    stockPrice.textContent = `$${parseFloat(quoteData['05. price']).toFixed(2)}`;
    const change = parseFloat(quoteData['09. change']);
    const changePercent = quoteData['10. change percent'];
    stockChange.innerHTML = `${change >= 0 ? '&#9650;' : '&#9660;'} ${changePercent}`;
    stockChange.className = `stock-change ${change >= 0 ? 'positive' : 'negative'}`;
    stockVolume.textContent = quoteData['06. volume'];
  }
  
  function showError(message) {
    errorContainer.textContent = message;
    errorContainer.style.display = 'block';
    resultContainer.style.display = 'none';
  }
});