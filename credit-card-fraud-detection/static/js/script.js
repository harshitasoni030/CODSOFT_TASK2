const form = document.getElementById("fraud-form");
const resultBox = document.getElementById("result");
const errorBox = document.getElementById("error");
const submitBtn = document.getElementById("submit-btn");

// A real row taken from fraudTest.csv (first row)
const EXAMPLE = {
  trans_date_trans_time: "2020-06-21T12:14",
  amt: 2.86, category: "personal_care", gender: "M", state: "SC",
  city_pop: 333497, lat: 33.9659, long: -80.9355,
  merch_lat: 33.986391, merch_long: -81.200714, dob: "1968-03-19",
};

document.getElementById("example-btn").addEventListener("click", () => {
  for (const [key, value] of Object.entries(EXAMPLE)) {
    if (form.elements[key]) form.elements[key].value = value;
  }
});

form.addEventListener("reset", () => {
  resultBox.classList.add("hidden");
  errorBox.classList.add("hidden");
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  errorBox.classList.add("hidden");
  resultBox.classList.add("hidden");

  if (!form.checkValidity()) {
    form.reportValidity();
    return;
  }

  const payload = Object.fromEntries(new FormData(form).entries());
  submitBtn.disabled = true;
  submitBtn.textContent = "Checking...";

  try {
    const response = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Prediction failed");
    showResult(data);
  } catch (err) {
    errorBox.textContent = err.message;
    errorBox.classList.remove("hidden");
  } finally {
    submitBtn.disabled = false;
    submitBtn.textContent = "Check Transaction";
  }
});

function showResult(data) {
  const percent = (data.fraud_probability * 100).toFixed(2);
  const isFraud = data.prediction === "FRAUDULENT";
  resultBox.className = "result " + (isFraud ? "fraud" : "legit");
  document.getElementById("result-label").textContent = data.prediction;
  document.getElementById("result-prob").textContent = "Fraud probability: " + percent + "%";
  document.getElementById("bar-fill").style.width = percent + "%";
}
