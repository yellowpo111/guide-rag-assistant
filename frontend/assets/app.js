"use strict";

const MAX_QUESTION_CHARACTERS = 2000;
const STATUS_LABELS = {
  accepted: { label: "Guard 通过", state: "accepted" },
  fallback_to_original: { label: "Guard 已回退原问题", state: "fallback" },
  unguarded: { label: "未启用 Guard", state: "neutral" },
};

const form = document.querySelector("#ask-form");
const questionInput = document.querySelector("#question");
const sendButton = document.querySelector("#send-button");
const conversation = document.querySelector("#conversation");
const emptyState = document.querySelector("#empty-state");
const characterCount = document.querySelector("#character-count");
const serviceStatus = document.querySelector("#service-status");
const serviceStatusText = document.querySelector("#service-status-text");
let activeRequestController = null;

form.addEventListener("submit", handleSubmit);
questionInput.addEventListener("input", updateCharacterCount);
questionInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    form.requestSubmit();
  }
});

checkReadiness();
questionInput.focus();

async function checkReadiness() {
  setServiceStatus("checking", "正在检查服务");
  try {
    const response = await fetch("/health/ready", {
      headers: { Accept: "application/json" },
    });
    if (!response.ok) {
      throw new Error("Service not ready");
    }
    setServiceStatus("ready", "服务就绪");
  } catch (_error) {
    setServiceStatus("unavailable", "服务不可用");
  }
}

async function handleSubmit(event) {
  event.preventDefault();
  if (activeRequestController) {
    activeRequestController.abort();
    return;
  }

  const question = questionInput.value.trim();
  if (!question) {
    return;
  }

  emptyState?.remove();
  appendUserTurn(question);
  const loadingTurn = appendLoadingTurn();
  const controller = new AbortController();
  activeRequestController = controller;
  let assistantUi = null;
  let completed = false;
  setSubmitting(true);
  scrollToLatest();

  try {
    const response = await fetch("/v1/assistant/stream", {
      method: "POST",
      headers: {
        Accept: "text/event-stream",
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ question }),
      signal: controller.signal,
    });

    if (!response.ok) {
      const payload = await parseJsonResponse(response);
      loadingTurn.remove();
      appendErrorTurn(errorMessageFor(response.status, payload), requestIdFrom(payload, response));
      return;
    }

    loadingTurn.remove();
    assistantUi = appendStreamingAssistantTurn(response.headers.get("X-Request-ID"));
    questionInput.value = "";
    updateCharacterCount();
    await consumeSse(response, (eventName, payload) => {
      if (eventName === "start") {
        setRequestId(assistantUi.turn, payload.request_id);
        assistantUi.requestId = payload.request_id || assistantUi.requestId;
      } else if (eventName === "route") {
        assistantUi.turn.dataset.route = payload.route || "unknown";
      } else if (eventName === "trace") {
        renderTrace(assistantUi, payload);
      } else if (eventName === "delta") {
        assistantUi.answer.textContent += typeof payload.text === "string" ? payload.text : "";
        scrollToLatest();
      } else if (eventName === "done") {
        completed = true;
        showFeedback(assistantUi);
      } else if (eventName === "error") {
        throw new StreamEventError(payload);
      }
    });
    if (!completed) {
      throw new Error("Stream ended before the done event.");
    }
    setServiceStatus("ready", "服务就绪");
  } catch (error) {
    loadingTurn.remove();
    if (error?.name === "AbortError") {
      if (assistantUi) {
        appendTurnStatus(assistantUi.turn, "已停止生成");
      }
      setServiceStatus("ready", "服务就绪");
    } else if (error instanceof StreamEventError) {
      const requestId = error.payload?.request_id || null;
      if (assistantUi) {
        appendInlineError(
          assistantUi.turn,
          "模型服务暂时不可用，回答未能完整生成。",
          requestId,
        );
      } else {
        appendErrorTurn("模型服务暂时不可用，请稍后重试。", requestId);
      }
    } else {
      if (assistantUi) {
        appendInlineError(assistantUi.turn, "流式连接异常，回答可能不完整。", null);
      } else {
        appendErrorTurn("无法连接助手服务，请检查服务是否已启动。", null);
      }
      setServiceStatus("unavailable", "服务不可用");
    }
  } finally {
    if (activeRequestController === controller) {
      activeRequestController = null;
    }
    setSubmitting(false);
    questionInput.focus();
    scrollToLatest();
  }
}

async function parseJsonResponse(response) {
  try {
    return await response.json();
  } catch (_error) {
    return {};
  }
}

function errorMessageFor(status, payload) {
  const messages = {
    422: "问题为空、过长或格式无效，请修改后重试。",
    502: "模型服务暂时不可用，请稍后重试。",
    503: "RAG 服务尚未就绪，请稍后重试。",
  };
  return messages[status] || payload?.error?.message || "请求未能完成，请稍后重试。";
}

function requestIdFrom(payload, response) {
  return payload?.error?.request_id || response.headers.get("X-Request-ID");
}

function appendUserTurn(question) {
  const fragment = document.querySelector("#user-turn-template").content.cloneNode(true);
  fragment.querySelector(".user-question").textContent = question;
  conversation.append(fragment);
}

function appendLoadingTurn() {
  const fragment = document.querySelector("#loading-turn-template").content.cloneNode(true);
  const turn = fragment.querySelector(".loading-turn");
  conversation.append(fragment);
  return turn;
}

function appendStreamingAssistantTurn(requestId) {
  const fragment = document.querySelector("#assistant-turn-template").content.cloneNode(true);
  const turn = fragment.querySelector(".turn-assistant");
  const trace = turn.querySelector(".trace");
  trace.hidden = true;
  setRequestId(turn, requestId);
  conversation.append(fragment);
  const ui = {
    turn,
    answer: turn.querySelector(".answer-text"),
    trace,
    feedback: turn.querySelector(".feedback"),
    feedbackStatus: turn.querySelector(".feedback-status"),
    feedbackButtons: [...turn.querySelectorAll(".feedback-button")],
    requestId,
    selectedRating: null,
  };
  ui.feedbackButtons.forEach((button) => {
    button.addEventListener("click", () => updateFeedback(ui, button.dataset.rating));
  });
  return ui;
}

function showFeedback(ui) {
  if (ui.requestId) {
    ui.feedback.hidden = false;
  }
}

async function updateFeedback(ui, rating) {
  if (!ui.requestId || !["positive", "negative"].includes(rating)) {
    return;
  }
  const revoke = ui.selectedRating === rating;
  setFeedbackSaving(ui, true);
  ui.feedbackStatus.textContent = "";
  try {
    const response = await fetch(
      `/v1/assistant/feedback/${encodeURIComponent(ui.requestId)}`,
      revoke
        ? { method: "DELETE" }
        : {
            method: "PUT",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ rating }),
          },
    );
    if (!response.ok) {
      throw new Error("Feedback request failed");
    }
    ui.selectedRating = revoke ? null : rating;
    renderFeedbackSelection(ui);
    ui.feedbackStatus.textContent = revoke ? "评价已撤销" : "评价已保存";
  } catch (_error) {
    ui.feedbackStatus.textContent = "保存失败，请重试";
  } finally {
    setFeedbackSaving(ui, false);
  }
}

function setFeedbackSaving(ui, saving) {
  ui.feedbackButtons.forEach((button) => {
    button.disabled = saving;
  });
}

function renderFeedbackSelection(ui) {
  ui.feedbackButtons.forEach((button) => {
    button.setAttribute(
      "aria-pressed",
      String(button.dataset.rating === ui.selectedRating),
    );
  });
}

function renderTrace(ui, payload) {
  setText(ui.turn, ".retrieval-query", valueOrDash(payload.retrieval_query));
  setText(ui.turn, ".rewrite-query", valueOrDash(payload.rewrite_query));
  renderGuardStatus(ui.turn.querySelector(".guard-status"), payload.query_rewrite_status);
  renderConstraints(ui.turn.querySelector(".required-constraints"), payload.required_constraints);
  renderConstraints(ui.turn.querySelector(".missing-constraints"), payload.missing_constraints);
  renderSources(ui.turn.querySelector(".sources-list"), payload.sources);
  ui.trace.hidden = false;
}

function setRequestId(turn, requestId) {
  setText(turn, ".request-id", requestId ? `Request ${requestId}` : "");
}

function appendErrorTurn(message, requestId) {
  const article = document.createElement("article");
  article.className = "turn turn-assistant";

  const heading = document.createElement("div");
  heading.className = "assistant-heading";
  const mark = document.createElement("span");
  mark.className = "assistant-mark";
  mark.setAttribute("aria-hidden", "true");
  mark.textContent = "F";
  const labelBlock = document.createElement("div");
  const label = document.createElement("p");
  label.className = "turn-label";
  label.textContent = "财政业务智能助手";
  labelBlock.append(label);
  if (requestId) {
    const request = document.createElement("p");
    request.className = "request-id";
    request.textContent = `Request ${requestId}`;
    labelBlock.append(request);
  }
  heading.append(mark, labelBlock);

  const error = document.createElement("p");
  error.className = "error-message";
  error.textContent = message;
  article.append(heading, error);
  conversation.append(article);
}

function appendInlineError(turn, message, requestId) {
  const suffix = requestId ? ` Request ${requestId}` : "";
  const error = textElement("p", "error-message", `${message}${suffix}`);
  turn.append(error);
}

function appendTurnStatus(turn, message) {
  turn.append(textElement("p", "stream-status", message));
}

class StreamEventError extends Error {
  constructor(payload) {
    super(payload?.message || "Assistant stream failed");
    this.name = "StreamEventError";
    this.payload = payload;
  }
}

async function consumeSse(response, onEvent) {
  if (!response.body) {
    throw new Error("Streaming response body is unavailable.");
  }
  const reader = response.body.getReader();
  const decoder = new TextDecoder("utf-8");
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
    buffer = dispatchCompleteSseBlocks(buffer, onEvent);
    if (done) {
      break;
    }
  }
  if (buffer.trim()) {
    dispatchSseBlock(buffer, onEvent);
  }
}

function dispatchCompleteSseBlocks(buffer, onEvent) {
  const normalized = buffer.replaceAll("\r\n", "\n");
  const blocks = normalized.split("\n\n");
  const remainder = blocks.pop() || "";
  blocks.forEach((block) => dispatchSseBlock(block, onEvent));
  return remainder;
}

function dispatchSseBlock(block, onEvent) {
  if (!block.trim() || block.trimStart().startsWith(":")) {
    return;
  }
  let eventName = "message";
  const dataLines = [];
  block.split("\n").forEach((line) => {
    if (line.startsWith("event:")) {
      eventName = line.slice(6).trim();
    } else if (line.startsWith("data:")) {
      dataLines.push(line.slice(5).trimStart());
    }
  });
  if (dataLines.length === 0) {
    return;
  }
  onEvent(eventName, JSON.parse(dataLines.join("\n")));
}

function renderGuardStatus(element, status) {
  const presentation = STATUS_LABELS[status] || {
    label: status || "状态未知",
    state: "neutral",
  };
  element.textContent = presentation.label;
  element.dataset.state = presentation.state;
}

function renderConstraints(container, constraints) {
  const values = Array.isArray(constraints) ? constraints : [];
  if (values.length === 0) {
    container.className = `${container.className} empty-value`;
    container.textContent = "无";
    return;
  }

  const list = document.createElement("span");
  list.className = "constraint-list";
  values.forEach((value) => {
    const item = document.createElement("span");
    item.className = "constraint-item";
    item.textContent = value;
    list.append(item);
  });
  container.append(list);
}

function renderSources(container, sources) {
  const values = Array.isArray(sources) ? sources : [];
  if (values.length === 0) {
    const empty = document.createElement("p");
    empty.className = "empty-value";
    empty.textContent = "无引用来源";
    container.append(empty);
    return;
  }

  values.forEach((source) => {
    const item = document.createElement("article");
    item.className = "source-item";

    const heading = document.createElement("div");
    heading.className = "source-heading";
    heading.append(
      textElement("span", "source-rank", `#${valueOrDash(source.rank)}`),
      textElement("span", "source-name", valueOrDash(source.source)),
    );

    const metadata = document.createElement("div");
    metadata.className = "source-meta";
    metadata.append(
      textElement("span", "", `章节：${valueOrDash(source.section)}`),
      textElement("span", "", `小节：${valueOrDash(source.subsection)}`),
      textElement("span", "source-score", `Dense ${formatScore(source.dense_score)}`),
      textElement("span", "source-score", `Rerank ${formatScore(source.rerank_score)}`),
    );

    item.append(heading, metadata);
    container.append(item);
  });
}

function textElement(tagName, className, value) {
  const element = document.createElement(tagName);
  if (className) {
    element.className = className;
  }
  element.textContent = value;
  return element;
}

function setText(root, selector, value) {
  root.querySelector(selector).textContent = value;
}

function valueOrDash(value) {
  return value === null || value === undefined || value === "" ? "-" : String(value);
}

function formatScore(value) {
  const number = Number(value);
  return Number.isFinite(number) ? number.toFixed(4) : "-";
}

function setSubmitting(isSubmitting) {
  questionInput.disabled = isSubmitting;
  sendButton.dataset.mode = isSubmitting ? "stop" : "send";
  sendButton.textContent = isSubmitting ? "停止" : "发送";
  sendButton.setAttribute("aria-label", isSubmitting ? "停止生成" : "发送消息");
}

function setServiceStatus(state, text) {
  serviceStatus.dataset.state = state;
  serviceStatusText.textContent = text;
}

function updateCharacterCount() {
  characterCount.textContent = `${questionInput.value.length} / ${MAX_QUESTION_CHARACTERS}`;
}

function scrollToLatest() {
  window.requestAnimationFrame(() => {
    window.scrollTo({ top: document.body.scrollHeight, behavior: "smooth" });
  });
}
