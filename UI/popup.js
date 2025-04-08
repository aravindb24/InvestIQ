// Global object to track stock cards by symbol,
// and a variable to record which stock is currently displayed ephemerally.
const stockCards = {};
let ephemeralSymbol = null;

document.addEventListener('DOMContentLoaded', () => {
  const apiKey = '4NIFR0WSYEG706YI'; // Replace with your Alpha Vantage API key
  const searchBtn = document.getElementById('searchBtn');
  const stockSymbolInput = document.getElementById('stockSymbol');
  const errorContainer = document.getElementById('errorContainer');
  const resultsContainer = document.getElementById('resultsContainer');

  // Initialize: update news headlines.
  fetchNewsDataCached();

  searchBtn.addEventListener('click', () => fetchStockData());
  stockSymbolInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') fetchStockData();
  });

  // Fetch stock data from Alpha Vantage.
  function fetchStockData() {
    const symbol = stockSymbolInput.value.trim().toUpperCase();
    if (!symbol) {
      showError('Please enter a stock symbol.');
      return;
    }
    clearError();

    const url = `https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol=${symbol}&apikey=${apiKey}`;
    fetch(url)
      .then(response => response.json())
      .then(data => {
        if (data['Global Quote'] && Object.keys(data['Global Quote']).length > 0) {
          displayOrUpdateStockCard(data['Global Quote']);
        } else {
          showError('Invalid stock symbol or no data available.');
        }
      })
      .catch(error => {
        console.error('Error fetching stock data:', error);
        showError('Failed to fetch stock data. Please try again.');
      });
  }

  function showError(message) {
    errorContainer.textContent = message;
    errorContainer.style.display = 'block';
  }

  function clearError() {
    errorContainer.textContent = '';
    errorContainer.style.display = 'none';
  }

  // --- Watchlist Helpers ---

  // Returns an array from localStorage for watchlisted stocks.
  function getWatchlist() {
    const list = localStorage.getItem('watchlistStocks');
    return list ? JSON.parse(list) : [];
  }

  // Saves the watchlist back to localStorage.
  function saveWatchlist(list) {
    localStorage.setItem('watchlistStocks', JSON.stringify(list));
  }

  // Check if a stock is watchlisted.
  function isWatchlisted(symbol) {
    const watchlist = getWatchlist();
    return watchlist.includes(symbol);
  }

  // Toggle the watchlist status for a given symbol.
  function toggleWatchlist(symbol) {
    let watchlist = getWatchlist();
    if (watchlist.includes(symbol)) {
      // Remove from watchlist.
      watchlist = watchlist.filter(item => item !== symbol);
      saveWatchlist(watchlist);
      // Remove the card from the display.
      if (stockCards[symbol]) {
        resultsContainer.removeChild(stockCards[symbol]);
        delete stockCards[symbol];
      }
      // If this symbol was ephemeral, clear the ephemeral marker.
      if (ephemeralSymbol === symbol) {
        ephemeralSymbol = null;
      }
    } else {
      // Add to watchlist.
      watchlist.unshift(symbol);  // New watchlisted items go at the beginning.
      saveWatchlist(watchlist);
      // Update the card's toggle button to watchlisted state.
      if (stockCards[symbol]) {
        const btn = stockCards[symbol].querySelector('.toggle-button');
        btn.textContent = '−';
      }
      // If this stock was ephemeral, clear the ephemeral marker.
      if (ephemeralSymbol === symbol) {
        ephemeralSymbol = null;
      }
    }
  }

  // --- Stock Card Functions ---

  // Creates a new stock card element.
  function createStockCard(quoteData) {
    const symbol = quoteData['01. symbol'];
    const card = document.createElement('div');
    card.className = 'stock-card';
    card.id = `stock-${symbol}`;

    card.innerHTML = `
      <div class="stock-header">${symbol}</div>
      <div class="stock-price">$${parseFloat(quoteData['05. price']).toFixed(2)}</div>
      <div class="stock-change ${parseFloat(quoteData['09. change']) >= 0 ? 'positive' : 'negative'}">
        ${parseFloat(quoteData['09. change']) >= 0 ? '&#9650;' : '&#9660;'} ${quoteData['10. change percent']}
      </div>
      <div class="stock-volume">Volume: ${quoteData['06. volume']}</div>
    `;
    // Create the sleek toggle button.
    const btn = document.createElement('button');
    btn.className = 'toggle-button';
    btn.textContent = isWatchlisted(symbol) ? '−' : '+';
    btn.onclick = () => toggleWatchlist(symbol);
    card.appendChild(btn);

    return card;
  }

  // Updates or creates a stock card in the unified results container.
  function displayOrUpdateStockCard(quoteData) {
    const symbol = quoteData['01. symbol'];
    if (isWatchlisted(symbol)) {
      // For watchlisted stocks:
      if (stockCards[symbol]) {
        // Update existing card.
        updateCardContent(stockCards[symbol], quoteData);
      } else {
        // Create new card and append it (watchlisted cards are fixed in place).
        const newCard = createStockCard(quoteData);
        resultsContainer.appendChild(newCard);
        stockCards[symbol] = newCard;
      }
    } else {
      // For non-watchlisted (ephemeral) search results:
      // If an ephemeral card already exists and it's for a different symbol, remove it.
      if (ephemeralSymbol !== null && ephemeralSymbol !== symbol && stockCards[ephemeralSymbol]) {
        resultsContainer.removeChild(stockCards[ephemeralSymbol]);
        delete stockCards[ephemeralSymbol];
      }
      ephemeralSymbol = symbol;
      if (stockCards[symbol]) {
        // Update the existing card and move it to the top.
        updateCardContent(stockCards[symbol], quoteData);
        resultsContainer.insertBefore(stockCards[symbol], resultsContainer.firstChild);
      } else {
        // Create a new ephemeral card and prepend it.
        const newCard = createStockCard(quoteData);
        resultsContainer.insertBefore(newCard, resultsContainer.firstChild);
        stockCards[symbol] = newCard;
      }
    }
  }

  // Updates the content of an existing card with new quote data.
  function updateCardContent(card, quoteData) {
    const symbol = quoteData['01. symbol'];
    card.querySelector('.stock-header').textContent = symbol;
    card.querySelector('.stock-price').textContent = `$${parseFloat(quoteData['05. price']).toFixed(2)}`;
    const change = parseFloat(quoteData['09. change']);
    const changePercent = quoteData['10. change percent'];
    const changeDiv = card.querySelector('.stock-change');
    changeDiv.innerHTML = `${change >= 0 ? '&#9650;' : '&#9660;'} ${changePercent}`;
    changeDiv.className = `stock-change ${change >= 0 ? 'positive' : 'negative'}`;
    card.querySelector('.stock-volume').textContent = `Volume: ${quoteData['06. volume']}`;
    // Update the toggle button.
    const btn = card.querySelector('.toggle-button');
    btn.textContent = isWatchlisted(symbol) ? '−' : '+';
  }

  // End of DOMContentLoaded.

  // --- Existing News Headlines Code with caching remains unchanged ---
  fetchNewsDataCached();
});

function fetchNewsDataCached() {
  const newsContainer = document.getElementById('newsContainer');
  const cachedData = localStorage.getItem('newsData');
  const cachedTimestamp = localStorage.getItem('newsTimestamp');
  const now = new Date().getTime();
  const oneDay = 24 * 60 * 60 * 1000; // one day in ms

  if (cachedData && cachedTimestamp && (now - cachedTimestamp < oneDay)) {
    displayNewsData(JSON.parse(cachedData));
  } else {
    newsContainer.innerHTML = '<div class="loading-container"><div class="spinner"></div></div>';
    fetchNewsDataFromAPI();
  }
}

function fetchNewsDataFromAPI() {
  const newsContainer = document.getElementById('newsContainer');
  const apiUrl = 'http://localhost:5000/api/news';
  fetch(apiUrl)
    .then(response => response.json())
    .then(data => {
      localStorage.setItem('newsData', JSON.stringify(data));
      localStorage.setItem('newsTimestamp', new Date().getTime());
      displayNewsData(data);
    })
    .catch(error => {
      newsContainer.innerHTML = '<p>Failed to load news headlines.</p>';
      console.error('Error fetching news headlines:', error);
    });
}

function displayNewsData(data) {
  const newsContainer = document.getElementById('newsContainer');
  newsContainer.innerHTML = '';
  if (data.length === 0) {
    newsContainer.innerHTML = '<p>No news headlines available at the moment.</p>';
  } else {
    const headlines = data.slice(0, 100);
    headlines.forEach(item => {
      const newsItem = document.createElement('div');
      newsItem.className = 'news-item';
      newsItem.innerHTML = `
        <div class="news-headline">${item.headline}</div>
        <div class="news-score">Pressing Score: ${item.pressing_score}</div>
        <div class="news-link"><a href="${item.link}" target="_blank">Read Article</a></div>
      `;
      newsContainer.appendChild(newsItem);
    });
  }
}
