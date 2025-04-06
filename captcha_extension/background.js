chrome.runtime.onInstalled.addListener(() => {
  console.log("Captcha Resolver instalado.");
  chrome.storage.local.set({ modoAutomatico: false }); // Inicializa como desativado
});
