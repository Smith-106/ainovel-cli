package bootstrap

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/voocel/agentcore"
)

// streamStub 按脚本返回流式事件，并记录 Generate 是否被调用（用于断言回落）。
type streamStub struct {
	streamErr   error
	events      []agentcore.StreamEvent
	generateErr error
	generateHit *int64
}

func (m *streamStub) Generate(context.Context, []agentcore.Message, []agentcore.ToolSpec, ...agentcore.CallOption) (*agentcore.LLMResponse, error) {
	if m.generateHit != nil {
		atomic.AddInt64(m.generateHit, 1)
	}
	if m.generateErr != nil {
		return nil, m.generateErr
	}
	return &agentcore.LLMResponse{Message: agentcore.Message{
		Role:    agentcore.RoleAssistant,
		Content: []agentcore.ContentBlock{agentcore.TextBlock("fallback-ok")},
	}}, nil
}

func (m *streamStub) GenerateStream(context.Context, []agentcore.Message, []agentcore.ToolSpec, ...agentcore.CallOption) (<-chan agentcore.StreamEvent, error) {
	if m.streamErr != nil {
		return nil, m.streamErr
	}
	ch := make(chan agentcore.StreamEvent, len(m.events))
	for _, ev := range m.events {
		ch <- ev
	}
	close(ch)
	return ch, nil
}

func (m *streamStub) SupportsTools() bool { return true }

// 流式成功时直接组装 Done 终态，不应再调非流式（省一次请求、无重复计费）。
func TestSwappableGeneratePrefersStream(t *testing.T) {
	var hits int64
	stub := &streamStub{generateHit: &hits, events: []agentcore.StreamEvent{
		{Type: agentcore.StreamEventTextDelta, Delta: "hi"},
		{Type: agentcore.StreamEventDone, Message: agentcore.Message{
			Role:    agentcore.RoleAssistant,
			Content: []agentcore.ContentBlock{agentcore.TextBlock("hi")},
		}, StopReason: agentcore.StopReasonStop},
	}}
	sw := NewSwappableModel("p", "m", stub, nil)
	resp, err := sw.Generate(context.Background(), []agentcore.Message{agentcore.UserMsg("hi")}, nil)
	if err != nil {
		t.Fatalf("stream-first generate: %v", err)
	}
	if resp.Message.TextContent() != "hi" || resp.Message.StopReason != agentcore.StopReasonStop {
		t.Fatalf("组装结果不一致: %+v", resp.Message)
	}
	if atomic.LoadInt64(&hits) != 0 {
		t.Fatalf("流式成功不应回落非流式，Generate hits = %d", hits)
	}
}

// 流初始化失败回落非流式（流式不支持的端点行为不变）。
func TestSwappableGenerateFallsBackWhenStreamInitFails(t *testing.T) {
	var hits int64
	stub := &streamStub{generateHit: &hits, streamErr: errors.New("stream unsupported")}
	sw := NewSwappableModel("p", "m", stub, nil)
	resp, err := sw.Generate(context.Background(), []agentcore.Message{agentcore.UserMsg("hi")}, nil)
	if err != nil {
		t.Fatalf("fallback generate: %v", err)
	}
	if resp.Message.TextContent() != "fallback-ok" {
		t.Fatalf("应回落非流式结果，got %q", resp.Message.TextContent())
	}
	if atomic.LoadInt64(&hits) != 1 {
		t.Fatalf("回落应恰调一次非流式，hits = %d", hits)
	}
}

// 流中 Error 事件同样回落非流式。
func TestSwappableGenerateFallsBackOnStreamErrorEvent(t *testing.T) {
	var hits int64
	stub := &streamStub{generateHit: &hits, events: []agentcore.StreamEvent{
		{Type: agentcore.StreamEventError, Err: errors.New("mid-stream boom")},
	}}
	sw := NewSwappableModel("p", "m", stub, nil)
	resp, err := sw.Generate(context.Background(), []agentcore.Message{agentcore.UserMsg("hi")}, nil)
	if err != nil {
		t.Fatalf("fallback generate: %v", err)
	}
	if resp.Message.TextContent() != "fallback-ok" {
		t.Fatalf("应回落非流式结果，got %q", resp.Message.TextContent())
	}
}

// 网关按形态路由的回归：非流式 403（免费池拒绝站外）但流式 200 时，
// Swap 冒烟必须与真实调用同形态（经 SwappableModel 流式优先），不得误杀。
func TestSwapSmokeFollowsStreamFirstShape(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var stream bool
		if r.URL.Path == "/chat/completions" {
			// 读一小段 body 判断是否为流式请求。
			buf := make([]byte, 4096)
			n, _ := r.Body.Read(buf)
			stream = strings.Contains(string(buf[:n]), `"stream":true`)
		}
		w.Header().Set("Content-Type", "application/json")
		if !stream {
			w.WriteHeader(http.StatusForbidden)
			_, _ = w.Write([]byte(`{"error":{"message":"OpenCode's free tier can only be used from within OpenCode","type":"auth"}}`))
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte("data: {\"id\":\"stub\",\"object\":\"chat.completion.chunk\",\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"ok\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n"))
	}))
	defer srv.Close()

	cfg := Config{
		Provider:  "good",
		ModelName: "good-model",
		Providers: map[string]ProviderConfig{
			"good": {Type: "openai", APIKey: "k", BaseURL: srv.URL, Models: []ModelConfig{{Name: "good-model"}}},
			"gw":   {Type: "glm", APIKey: "k", BaseURL: srv.URL, Models: []ModelConfig{{Name: "gw-model"}}},
		},
	}
	cfg.FillDefaults()
	ms, err := NewModelSet(cfg)
	if err != nil {
		t.Fatalf("NewModelSet: %v", err)
	}
	if err := ms.Swap("default", "gw", "gw-model"); err != nil {
		t.Fatalf("流式可用时 Swap 不应被冒烟拦截，got: %v", err)
	}
	if provider, name := ms.Default.Current(); provider != "gw" || name != "gw-model" {
		t.Fatalf("Swap 后应为 gw/gw-model，got %s/%s", provider, name)
	}
}
