document.addEventListener("DOMContentLoaded", () => {
    const messagesContainer = document.getElementById("messagesContainer");
    const chatForm = document.getElementById("chatForm");
    const userInput = document.getElementById("userInput");
    const sendBtn = document.getElementById("sendBtn");
    const clearBtn = document.getElementById("clearBtn");
    const newChatBtn = document.getElementById("newChatBtn");
    const welcomeCard = document.getElementById("welcomeCard");

    const systemPromptInput = document.getElementById("systemPrompt");
    const tempSlider = document.getElementById("tempSlider");
    const tempValue = document.getElementById("tempValue");
    const maxTokensSlider = document.getElementById("maxTokensSlider");
    const maxTokensValue = document.getElementById("maxTokensValue");
    const hfTokenInput = document.getElementById("hfTokenInput");
    const statusText = document.getElementById("statusText");

    const HF_MODEL_ID = "TemurbekHamzaev/qwen2.5-1.5b-chatbot";
    const HF_ROUTER_URL = "https://router.huggingface.co/hf-inference/v1/chat/completions";

    // Load saved HF Token from localStorage
    if (hfTokenInput) {
        hfTokenInput.value = localStorage.getItem("hf_token") || "";
        hfTokenInput.addEventListener("input", (e) => {
            localStorage.setItem("hf_token", e.target.value.trim());
        });
    }

    // State
    let conversationHistory = [];
    let isGenerating = false;

    // Sliders UI sync
    tempSlider.addEventListener("input", (e) => tempValue.textContent = e.target.value);
    maxTokensSlider.addEventListener("input", (e) => maxTokensValue.textContent = e.target.value);

    // Auto-resize textarea
    userInput.addEventListener("input", () => {
        userInput.style.height = "auto";
        userInput.style.height = `${Math.min(userInput.scrollHeight, 150)}px`;
    });

    // Enter to send (Shift+Enter for newline)
    userInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            chatForm.dispatchEvent(new Event("submit"));
        }
    });

    // Prompt Chips
    document.querySelectorAll(".chip").forEach(chip => {
        chip.addEventListener("click", () => {
            userInput.value = chip.dataset.prompt;
            userInput.dispatchEvent(new Event("input"));
            chatForm.dispatchEvent(new Event("submit"));
        });
    });

    // Clear Chat
    const resetChat = () => {
        conversationHistory = [];
        messagesContainer.innerHTML = "";
        if (welcomeCard) messagesContainer.appendChild(welcomeCard);
        userInput.focus();
    };

    clearBtn.addEventListener("click", resetChat);
    newChatBtn.addEventListener("click", resetChat);

    // Append a message bubble to the container
    const appendMessage = (role, text = "") => {
        if (welcomeCard && welcomeCard.parentNode === messagesContainer) {
            welcomeCard.remove();
        }

        const row = document.createElement("div");
        row.className = `message-row ${role}`;

        const avatar = document.createElement("div");
        avatar.className = "avatar";
        avatar.textContent = role === "user" ? "U" : "AI";

        const bubble = document.createElement("div");
        bubble.className = "bubble";
        bubble.innerHTML = marked.parse(text);

        row.appendChild(avatar);
        row.appendChild(bubble);
        messagesContainer.appendChild(row);
        messagesContainer.scrollTop = messagesContainer.scrollHeight;

        return bubble;
    };

    // Chat Form Submission
    chatForm.addEventListener("submit", async (e) => {
        e.preventDefault();
        const prompt = userInput.value.trim();
        if (!prompt || isGenerating) return;

        isGenerating = true;
        sendBtn.disabled = true;
        userInput.value = "";
        userInput.style.height = "auto";

        // 1. Add User Message
        appendMessage("user", prompt);
        conversationHistory.push({ role: "user", content: prompt });

        // 2. Prepare Assistant Bubble with Typing Cursor
        const assistantBubble = appendMessage("assistant", "");
        const cursor = document.createElement("span");
        cursor.className = "typing-cursor";
        assistantBubble.appendChild(cursor);

        let fullReply = "";

        // 3. Build Full Messages Payload with System Prompt
        const payloadMessages = [
            { role: "system", content: systemPromptInput.value.trim() },
            ...conversationHistory
        ];

        const token = (hfTokenInput ? hfTokenInput.value.trim() : "") || localStorage.getItem("hf_token") || "";

        try {
            // Determine if running locally with FastAPI or standalone on HF Spaces
            const isLocalBackend = window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1";
            let response;

            if (isLocalBackend) {
                // Local FastAPI SSE Stream
                response = await fetch("/api/chat/stream", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        messages: payloadMessages,
                        max_tokens: parseInt(maxTokensSlider.value),
                        temperature: parseFloat(tempSlider.value)
                    })
                });
            } else {
                // Hugging Face Router Serverless Streaming API
                const headers = { "Content-Type": "application/json" };
                if (token) {
                    headers["Authorization"] = `Bearer ${token}`;
                }

                response = await fetch(HF_ROUTER_URL, {
                    method: "POST",
                    headers: headers,
                    body: JSON.stringify({
                        model: HF_MODEL_ID,
                        messages: payloadMessages,
                        max_tokens: parseInt(maxTokensSlider.value),
                        temperature: parseFloat(tempSlider.value),
                        stream: true
                    })
                });
            }

            if (!response.ok) {
                if (response.status === 401 || response.status === 403) {
                    throw new Error("Hugging Face authorization required. Please paste your free HF Access Token in the sidebar settings.");
                } else if (response.status === 503) {
                    throw new Error("Model is currently loading onto Hugging Face serverless hardware. Please wait ~15 seconds and send again!");
                } else {
                    const errBody = await response.text();
                    throw new Error(`HTTP ${response.status}: ${errBody.slice(0, 150)}`);
                }
            }

            // Read SSE stream
            const reader = response.body.getReader();
            const decoder = new TextDecoder("utf-8");
            let buffer = "";

            while (true) {
                const { value, done } = await reader.read();
                if (done) break;

                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split("\n");
                buffer = lines.pop(); // keep last unfinished line

                for (const line of lines) {
                    const trimmed = line.trim();
                    if (!trimmed || trimmed.startsWith(":")) continue;

                    if (trimmed.startsWith("data:")) {
                        const rawData = trimmed.replace("data:", "").trim();
                        if (rawData === "[DONE]") break;

                        try {
                            const data = JSON.parse(rawData);
                            // Support both local token format and OpenAI/HF delta format
                            const tokenDelta = data.token || data.choices?.[0]?.delta?.content || "";
                            if (tokenDelta) {
                                fullReply += tokenDelta;
                                assistantBubble.innerHTML = marked.parse(fullReply);
                                assistantBubble.appendChild(cursor);
                                messagesContainer.scrollTop = messagesContainer.scrollHeight;
                            }
                        } catch (err) {
                            // ignore partial JSON parse error
                        }
                    }
                }
            }

        } catch (err) {
            fullReply += `\n\n⚠️ *${err.message}*`;
            assistantBubble.innerHTML = marked.parse(fullReply);
        } finally {
            cursor.remove();
            // Highlight syntax in newly rendered code blocks
            assistantBubble.querySelectorAll("pre code").forEach((el) => {
                hljs.highlightElement(el);
            });
            conversationHistory.push({ role: "assistant", content: fullReply });
            isGenerating = false;
            sendBtn.disabled = false;
            userInput.focus();
        }
    });
});
