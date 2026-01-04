# Review Sentiment Analyzer

## One-line Description

A backend text classification system for user reviews built with a clean API architecture.

---

## Tech Stack

* **Language:** Python
* **Framework:** Flask *(or FastAPI)*
* **Machine Learning:** scikit-learn
* **API Style:** REST API
* **Containerization:** Docker

---

## What It Is

The **Review Sentiment Analyzer** is a backend service that ingests user-written text reviews and returns a sentiment label — **positive** or **negative** — through a simple and clean REST API. It is designed to demonstrate practical machine learning deployment with a production-ready API structure.

---

## Key Features

* 🔹 REST API endpoint: `/predict`
* 🔹 Clean and modular model training script
* 🔹 Pre-trained sentiment classification model
* 🔹 Dockerized application for easy deployment
* 🔹 Example API request and response

---

## API Usage

### Endpoint

```
POST /predict
```

### Example Request

```json
{
  "review": "The product quality is excellent and delivery was fast"
}
```

### Example Response

```json
{
  "sentiment": "positive"
}
```

---

## Project Structure

```
ML/
├── app.py                # API entry point
├── model.py              # Model training and loading logic
├── requirements.txt      # Python dependencies
├── Dockerfile             # Docker configuration
├── README.md              # Project documentation
└── data/                  # Dataset (if applicable)
```

---

## How to Run

### Clone the Repository

```bash
git clone https://github.com/yash25raj/ML
cd ML
```

### Build Docker Image

```bash
docker build -t sentiment-api .
```

### Run the Container

```bash
docker run -p 5000:5000 sentiment-api
```

The API will be available at:

```
http://localhost:5000
```

---

## Use Cases

* Product review analysis
* Customer feedback monitoring
* Opinion mining systems
* NLP backend service demonstrations

---

## Future Enhancements

* Support for multi-class sentiment (neutral, mixed)
* Model performance evaluation metrics
* Database integration
* Frontend interface
* CI/CD pipeline integration

---

## Author

**Yash Raj**

---

⭐ If you find this project useful, consider giving it a star on GitHub!
