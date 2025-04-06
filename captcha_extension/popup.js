const checkbox = document.getElementById("modo-automatico");
const manualSection = document.getElementById("manual-section");
const botaoMirar = document.getElementById("mirar");

botaoMirar.addEventListener("click", () => {
  chrome.tabs.query({ active: true, currentWindow: true }, function (tabs) {
    chrome.tabs.sendMessage(tabs[0].id, { acao: "ativarMira" });
  });
});


document.addEventListener("DOMContentLoaded", () => {
  chrome.storage.local.get("modoAutomatico", (data) => {
    const ativado = !!data.modoAutomatico; // Garante que seja boolean
    checkbox.checked = ativado;
    manualSection.style.display = ativado ? "none" : "block";
  });
});

