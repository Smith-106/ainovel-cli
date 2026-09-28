package bootstrap

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/voocel/agentcore"
	"github.com/voocel/agentcore/llm"
)

// 回归：CPA-GLM 这类 OpenAI 兼容代理曾配成 type=openai；全局 reasoning_effort
// （如 xhigh）经 loop/arbiter 下发为显式 thinking 后，litellm-go openai
// provider 按模型名做本地门控（仅 gpt-5+ 算推理模型，request.go
// isReasoningModel），muse-spark-1.3-contributor-free 这类名字直接本地报错
// "thinking is only supported for reasoning chat models"，请求发不出去——
// 用户在 pi(cpa-responses) 里明明可用，ainovel 侧却必挂（goal #39）。
func TestOpenaiTypedProxyModelRejectsExplicitThinking(t *testing.T) {
	var hits int
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hits++
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"stub","object":"chat.completion","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`))
	}))
	defer srv.Close()

	pc := ProviderConfig{Type: "openai", APIKey: "k", BaseURL: srv.URL}
	m, err := createModelFromConfig("CPA-GLM", "muse-spark-1.3-contributor-free", pc, make(map[string]agentcore.ChatModel))
	if err != nil {
		t.Fatalf("create model: %v", err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	_, err = m.Generate(ctx, []agentcore.Message{agentcore.UserMsg("hi")}, nil,
		agentcore.WithMaxTokens(64), agentcore.WithThinking(agentcore.ThinkingXHigh))
	if err == nil || !strings.Contains(err.Error(), "thinking is only supported") {
		t.Fatalf("openai 协议+显式 thinking 应本地门控报错，got: %v", err)
	}
	if hits != 0 {
		t.Fatalf("本地门控应先于网络生效，server hits = %d", hits)
	}
}

// 修复：muse-spark 系是智谱 GLM 家族，走 type=glm 后 thinking 经厂商映射
// （thinking:{type:enabled} + reasoning_effort）出站，与 pi(cpa-responses)
// 的成功形状一致，不再被本地门控误杀。
func TestGlmTypedProxyModelForwardsThinkingAsReasoningEffort(t *testing.T) {
	var body map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(raw, &body)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"stub","object":"chat.completion","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`))
	}))
	defer srv.Close()

	pc := ProviderConfig{Type: "glm", APIKey: "k", BaseURL: srv.URL}
	m, err := createModelFromConfig("CPA-GLM", "muse-spark-1.3-contributor-free", pc, make(map[string]agentcore.ChatModel))
	if err != nil {
		t.Fatalf("create model: %v", err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	resp, err := m.Generate(ctx, []agentcore.Message{agentcore.UserMsg("hi")}, nil,
		agentcore.WithMaxTokens(64), agentcore.WithThinking(agentcore.ThinkingXHigh))
	if err != nil {
		t.Fatalf("glm 协议+显式 thinking 应正常出站，got: %v", err)
	}
	if resp == nil || resp.Message.TextContent() != "ok" {
		t.Fatalf("响应解析异常: %+v", resp)
	}
	if got, _ := body["reasoning_effort"].(string); got != "xhigh" {
		t.Fatalf("body.reasoning_effort = %q, want xhigh (body=%v)", got, body)
	}
	thinking, _ := body["thinking"].(map[string]any)
	if thinking == nil || thinking["type"] != "enabled" {
		t.Fatalf("body.thinking 应为 {type:enabled} (body=%v)", body)
	}
}

// glm 能力声明 Thinking.Supported != No，ResolveThinkingForModel 不得把已配
// 的 xhigh 钳成 auto——否则 /model 面板与下发意图被静默改写。
func TestGlmTypedModelKeepsConfiguredThinkingLevel(t *testing.T) {
	m, err := llm.NewModel("glm", "muse-spark-1.3-contributor-free",
		llm.WithAPIKey("k"), llm.WithBaseURL("https://example.com/v1"))
	if err != nil {
		t.Fatalf("new glm model: %v", err)
	}
	if got := m.Capabilities().Thinking.Supported; got == llm.SupportNo {
		t.Fatalf("glm thinking 支持度不应为 No，got %v", got)
	}
	resolved, ok := llm.ThinkingPolicyFor(m).Resolve(agentcore.ThinkingXHigh)
	if !ok || resolved != agentcore.ThinkingXHigh {
		t.Fatalf("xhigh 应被保留，got %q ok=%v", resolved, ok)
	}
}
