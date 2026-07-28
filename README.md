# 🎬 MovieIQ Dashboard

**Predictive Analytics on Film Success using Machine Learning**

MovieIQ is an interactive web dashboard built with **Streamlit** and **Scikit-Learn** that predicts whether a film will be a box office success or failure based on its budget, popularity, runtime, and viewer ratings.

## ✨ Features

- **📊 Interactive Analytics Dashboard**: View insightful charts, including a success/failure distribution pie chart, a Budget vs Revenue scatter plot, and a genre distribution breakdown.
- **🤖 Machine Learning Predictor**: Utilizes a pre-trained **Random Forest Classifier** (with 87.4% accuracy) to predict if a movie's revenue will exceed its budget.
- **🎯 Dynamic Filtering**: Filter the dataset in real-time by Minimum Rating and Genre.
- **📄 Data Explorer**: Browse through the underlying movie dataset directly from the UI.
- **🎨 Premium Dark Theme UI**: A custom, modern, and sleek design using CSS gradients and animations.

## 🛠️ Tech Stack

- **Python** 🐍
- **Streamlit** (Frontend/UI)
- **Scikit-Learn** (Machine Learning Model)
- **Pandas** & **NumPy** (Data Processing)
- **Matplotlib** & **Seaborn** (Data Visualization)

## 🚀 How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone https://github.com/danduboyinagurulakshmi/moviepw.git
   cd moviepw
   ```

2. **Install the required dependencies:**
   ```bash
   pip install streamlit pandas matplotlib seaborn scikit-learn scipy joblib
   ```

3. **Run the Streamlit application:**
   ```bash
   python -m streamlit run app.py
   ```

4. **View the Dashboard:**
   Open your browser and navigate to `http://localhost:8501`.

## 📁 Project Structure

- `app.py`: The main Streamlit web application.
- `MovieIQ.py`: The data processing and model training script.
- `movie_model.pkl`: The saved, pre-trained Random Forest model.
- `movies.csv`: The dataset containing movie information (budget, revenue, popularity, runtime, ratings, genres).
- `*.png`: Various charts generated during the model training phase.

---
*Built with ❤️ using Python and Streamlit.*