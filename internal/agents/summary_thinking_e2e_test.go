package agents

import (
	"context"
	"errors"
	"strconv"
	"strings"
	"testing"

	"github.com/voocel/agentcore"
	corecontext "github.com/voocel/agentcore/context"
)

// openaiLikeModel 模拟 litellm openai provider 的本地校验：非推理模型遇到显式
// thinking 设置直接报错（litellm/provider/openai/request.go），请求发不出去。
type openaiLikeModel struct {
	stubSummaryModel
	levels []agentcore.ThinkingLevel
}

func (m *openaiLikeModel) Generate(ctx context.Context, msgs []agentcore.Message, tools []agentcore.ToolSpec, opts ...agentcore.CallOption) (*agentcore.LLMResponse, error) {
	lv := agentcore.ResolveCallConfig(opts).ThinkingLevel
	m.levels = append(m.levels, lv)
	if lv != "" {
		return nil, errors.New("openai: thinking is only supported for reasoning chat models")
	}
	return m.stubSummaryModel.Generate(ctx, msgs, tools, opts...)
}

func TestSummaryWrapperEndToEndAgainstOpenaiValidation(t *testing.T) {
	msgs := []agentcore.AgentMessage{agentcore.UserMsg("生成第 1 卷卷摘要")}
	msgs = append(msgs, toolCallResult("ctx", "novel_context", `{}`, strings.Repeat("x", 8000))...)
	for i := 1; i <= 8; i++ {
		chapter := strconv.Itoa(i)
		msgs = append(msgs, toolCallResult("ch"+chapter, "read_chapter", `{"chapter":`+chapter+`}`, strings.Repeat("章", 4000))...)
	}

	// 未包装（直调库 Engine，绕过 newContextManager 的包装）：复现截图故障——压缩失败。
	raw := &openaiLikeModel{}
	sc := editorContextProfile.Summary
	sc.Model = raw
	rawEngine := corecontext.NewEngine(corecontext.EngineConfig{
		ContextWindow:    32000,
		ReserveTokens:    8000,
		CommitStrategies: []string{"full_summary"},
		Strategies:       []corecontext.Strategy{corecontext.NewFullSummary(sc)},
	})
	_, err := rawEngine.Project(context.Background(), msgs)
	if err == nil || !strings.Contains(err.Error(), "thinking is only supported") {
		t.Fatalf("未包装时应复现 thinking 校验失败，got: %v", err)
	}

	// newContextManager 内部自动包装：压缩成功。
	fixed := &openaiLikeModel{}
	projection, err := newRoleContextManager(editorContextProfile, fixed, 32000, "novel_context").Project(context.Background(), msgs)
	if err != nil {
		t.Fatalf("包装后压缩应成功，got: %v", err)
	}
	if !projection.ShouldCommit {
		t.Fatal("expected commit after successful compaction")
	}
	for _, lv := range fixed.levels {
		if lv != "" {
			t.Fatalf("内层模型不应再看到显式 thinking，got %q", lv)
		}
	}
}
