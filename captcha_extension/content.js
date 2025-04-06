async function encontrarCaptchaResolverEInserir() {
  const img = document.querySelector(".sat-captcha-image-container img");
  const input = document.querySelector("input[placeholder='Digite o texto']");

  if (!img || !input) {
    console.error("Imagem ou input não encontrados.");
    return;
  }

  try {
    const imagemBase64 = await converterImagemParaBase64(img);

    const resposta = await fetch("http://localhost:5000/resolver-captcha", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({ imagem: imagemBase64 })
    });

    const json = await resposta.json();
    const resultado = json.resultado;

    if (resultado) {
      input.value = resultado;
      console.log("CAPTCHA resolvido:", resultado);
    } else {
      alert("Não foi possível resolver o captcha.");
    }

  } catch (erro) {
    console.error("Erro ao resolver captcha:", erro);
    alert("Erro ao resolver captcha.");
  }
}

function converterImagemParaBase64(imgElement) {
  return new Promise((resolve, reject) => {
    try {
      const canvas = document.createElement("canvas");
      canvas.width = imgElement.naturalWidth;
      canvas.height = imgElement.naturalHeight;
      const ctx = canvas.getContext("2d");
      ctx.drawImage(imgElement, 0, 0);
      const dataURL = canvas.toDataURL("image/png");
      resolve(dataURL.replace(/^data:image\/(png|jpg);base64,/, ""));
    } catch (err) {
      reject(err);
    }
  });
}

// Escuta o botão "mirar" da popup
chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  if (msg.acao === "ativarMira") {
    encontrarCaptchaResolverEInserir();
  }
});

// Sempre executa ao carregar
encontrarCaptchaResolverEInserir();


function observarErroCaptcha() {
  const observer = new MutationObserver(() => {
    const erroCaptcha = document.querySelector("#__SatMessageBox .sat-vs-error");
    if (erroCaptcha && erroCaptcha.innerText.includes("Os caracteres digitados não coincidem")) {
      console.log("Erro de captcha detectado. Reexecutando resolução...");

      // Tenta resolver novamente
      encontrarCaptchaResolverEInserir();
    }
  });

  observer.observe(document.body, { childList: true, subtree: true });
}

// Inicia o observador de erro
observarErroCaptcha();

