# Time Series Labeling Tool

This is an open-source tool for labeling prices in time series data. You can label one or two prices for each time series. Additionally, you can navigate through the charts (by press arrows ← →) and find similar time series using dynamic time warping (DTW) by pressing the 'd' key when you're unsure about labeling a particular chart.

## Demo

Here’s a short demo of how the tool works:

![Demo](docs/images/time_series_labeling_demo.gif)

*P.S. I've used this code in the labeling of tens of thousands of time series. Now in prod I am using a newer version (plotly.js + fastapiб which is more flexible faster and has a lot of cool tools), maybe I will find time and a way to put this version into open source as well.*

## Instructions

After launching you will go to a simple settings menu, after that you can start labeling!


- If labeling 1 price, click in the desired place on the plot (the price you want to set) as `labeled_price_1`.
- If labeling 2 prices, click once for `labeled_price_1` and click again for `labeled_price_2`.
*see `Number of prices to label` in the settings menu at launch*
- Press 'd' to find similar time series using DTW (Euclidean distance).
- use the backward and forward arrows (← →) to move between the plots

*The labeled data will be saved in the `datasets/in_process_of_labeling` folder.*

## Installation

You can install the necessary dependencies in two ways:

### Option 1: Using `pipenv`

1. Install `pipenv` if you don't have it already:

   ```bash
   pip install pipenv
   ```

2. Install dependencies from the `Pipfile`:

   ```bash
   pipenv install
   ```

3. To activate the virtual environment, run:

   ```bash
   pipenv shell
   ```

4. Run the application:

   ```bash
   python app.py
   ``` 
   or
   ```bash
   pipenv run python app.py
   ```

### Option 2: Using `pip` and `requirements.txt`


1. Install dependencies using `pip`:

   ```bash
   pip install -r requirements.txt
   ```

2. Run the application:

   ```bash
   python app.py
   ```

## Usage

1. Run the application with:

   ```bash
   python app.py
   ```

2. The application will open a window where you can label the prices of time series.

3. Once you've finished labeling, the labeled data will be saved in the `datasets/in_process_of_labeling` folder.



## Contributing

Feel free to fork the repository, open issues, and submit pull requests. Contributions are welcome!

---

### `requirements.txt` Example (generated via `pipenv`):

```txt
fastdtw==0.3.4
matplotlib==3.10.1
numpy==2.2.4 ; python_version >= '3.10'
pandas==2.2.3
pydantic==2.10.6
scipy==1.15.2
```

---

**Changelog**

* v1.0: Initial release

## TODO:

- [ ] change the input data format so that it is possible to label time series of any length (by specifying the desired size in the settings).

**Authors**

* [dr11m](https://github.com/dr11m)

**Copyright**

Copyright (c) 2023 [dr11m]. All rights reserved.