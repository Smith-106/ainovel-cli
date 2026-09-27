package bootstrap

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// 回归：用户切到代理端拒绝站外调用的免费模型后，请求在 40 秒 arbiter 超时才挂
// （OpenCode's free tier can only be used from within OpenCode [auth, HTTP 403]）。
// 修复：Swap 生效前对新模型做最小请求冒烟，失败当场拦截、保留原模型。
func TestSwapSmokeTestRejectsForbiddenModel(t *testing.T) {
	forbidden := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusForbidden)
		_, _ = w.Write([]byte(`{"error":{"message":"OpenCode's free tier can only be used from within OpenCode","type":"auth"}}`))
	}))
	defer forbidden.Close()

	healthy := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"stub","object":"chat.completion","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`))
	}))
	defer healthy.Close()

	cfg := Config{
		Provider:  "good",
		ModelName: "good-model",
		Providers: map[string]ProviderConfig{
			"good": {Type: "openai", APIKey: "k", BaseURL: healthy.URL, Models: []ModelConfig{{Name: "good-model"}}},
			"bad":  {Type: "openai", APIKey: "k", BaseURL: forbidden.URL, Models: []ModelConfig{{Name: "bad-model"}}},
		},
	}
	cfg.FillDefaults()

	ms, err := NewModelSet(cfg)
	if err != nil {
		t.Fatalf("NewModelSet: %v", err)
	}

	if err := ms.Swap("default", "bad", "bad-model"); err == nil {
		t.Fatal("切到 403 模型应被冒烟拦截")
	} else if !strings.Contains(err.Error(), "连通性检查失败") || !strings.Contains(err.Error(), "保留原模型") {
		t.Fatalf("错误信息应说明拦截+保留原模型，got: %v", err)
	}

	// 原模型必须保持：provider/model 与实例均未动。
	if provider, name := ms.Default.Current(); provider != "good" || name != "good-model" {
		t.Fatalf("Swap 失败后原模型应保持，got %s/%s", provider, name)
	}
	if got := ModelProvider(ms.Default); got != "good" {
		t.Fatalf("Default ModelProvider 应仍为 good，got %q", got)
	}
}
