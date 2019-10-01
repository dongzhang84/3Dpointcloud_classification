const sendRequest = config => {
  const { url = '', args = {}, method = 'POST' } = config
  let request
  if (method == 'POST') {
    request = fetch(url, {
      method: 'POST', // *GET, POST, PUT, DELETE, etc.
      headers: {
        'Content-Type': 'application/json; charset=utf-8',
      },
      body: JSON.stringify(args),
    })
  } else if (method == 'GET') {
    const queryString = `${url}?` + Object.entries(args)
      .map(([key,value]) => `${key}=${value}`)
      .join('&')

    request = fetch(queryString, {
      method: 'GET',
      //mode: 'same-origin',
      //credentials: 'same-origin'
    })
  }
  return request.then(response => {console.log(response); return response.json()})
}

function updateBookRecommendation(itemID) {
  const rowNum = Number(document.getElementById(itemID).id.split('b')[1])
  const asin = document.getElementById(itemID).closest("tr").id

  const recSign = itemID[0]
  if (recSign == 'p') {recN = 5.0} else {recN = 1.0}

  sendRequest({
    url: 'http://127.0.0.1:5000/receivedInfo',
    args: {
      id: asin,
      recVal: recN,
      type: 'book'
    },
    method: 'GET'
  }).then(d => {
    console.log(d)
    const img = document.getElementById('book_img' + rowNum)
    const title = document.getElementById('book_title' + rowNum)
    const itemLink = document.getElementById('book_link' + rowNum)
    const rowInfo = document.getElementById(asin)
    img.setAttribute("src", d.img_url)
    itemLink.setAttribute("href", d.product_url)
    rowInfo.setAttribute("id", d.asin)
    title.textContent = d.title
  })
}

function updateMovieRecommendation(itemID) {
  const rowNum = Number(document.getElementById(itemID).id.split('m')[1])
  const asin = document.getElementById(itemID).closest("tr").id

  const recSign = itemID[0]
  if (recSign == 'p') {recN = 5.0} else {recN = 1.0}

  sendRequest({
    url: 'http://127.0.0.1:5000/receivedInfo',
    args: {
      id: asin,
      recVal: recN,
      type: 'movie'
    },
    method: 'GET'
  }).then(d => {
    console.log(d)
    const img = document.getElementById('movie_img' + rowNum)
    const title = document.getElementById('movie_title' + rowNum)
    const itemLink = document.getElementById('movie_link' + rowNum)
    const rowInfo = document.getElementById(asin)
    img.setAttribute("src", d.img_url)
    itemLink.setAttribute("href", d.product_url)
    rowInfo.setAttribute("id", d.asin)
    title.textContent = d.title
  })
}