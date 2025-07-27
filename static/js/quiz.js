document.getElementById("quiz-form").addEventListener("submit", function (e) {
  e.preventDefault();

  const form = e.target;
  const personality = [
    parseInt(form.q1.value),
    parseInt(form.q2.value),
    parseInt(form.q3.value),
    parseInt(form.q4.value)
  ];

  fetch("/predict", {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({ personality })
  })
    .then(res => res.json())
    .then(data => {
      if (data.redirect) {
        window.location.href = data.redirect; // ✅ redirect to result page
      } else {
        alert("Unexpected response format.");
      }
    })
    .catch(err => {
      console.error("Error:", err);
      alert("⚠️ Something went wrong while fetching the recommendation.");
    });
});
