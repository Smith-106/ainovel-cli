package agents

import (
	"context"
	"slices"

	"github.com/voocel/agentcore"
)

// summaryModel 包装 FullSummary 摘要专用的 ChatModel，剥离 ThinkingOff。
//
// 背景：agentcore 的 standalone 摘要路径（主摘要 generateSummary / turn 前缀
// generateTurnPrefixSummary）硬编码 WithThinking(ThinkingOff)；而 litellm 的
// openai provider 对非推理模型遇到显式 off 直接本地报错
// "openai: thinking is only supported for reasoning chat models"，请求根本发不出去，
// 且 turn 前缀摘要无重试——于是每次压缩必挂（project context: compaction turn prefix）。
// 非推理模型本就没有 thinking 概念，unset 与 off 等价，因此 resolve 出 off 时追加
// unset 覆盖即可。仅用于摘要模型（sc.Model），主循环 thinking 语义不受影响。
type summaryModel struct {
	inner agentcore.ChatModel
}

// wrapSummaryModel 为摘要路径包装模型；nil 保持 nil。
func wrapSummaryModel(model agentcore.ChatModel) agentcore.ChatModel {
	if model == nil {
		return nil
	}
	if _, ok := model.(*summaryModel); ok {
		return model
	}
	return &summaryModel{inner: model}
}

// stripThinkingOff resolve 出 off 时追加 unset 覆盖；其余原样透传。
func stripThinkingOff(opts []agentcore.CallOption) []agentcore.CallOption {
	if agentcore.ResolveCallConfig(opts).ThinkingLevel != agentcore.ThinkingOff {
		return opts
	}
	out := slices.Clone(opts)
	return append(out, agentcore.WithThinking(""))
}

func (m *summaryModel) Generate(ctx context.Context, messages []agentcore.Message, tools []agentcore.ToolSpec, opts ...agentcore.CallOption) (*agentcore.LLMResponse, error) {
	return m.inner.Generate(ctx, messages, tools, stripThinkingOff(opts)...)
}

func (m *summaryModel) GenerateStream(ctx context.Context, messages []agentcore.Message, tools []agentcore.ToolSpec, opts ...agentcore.CallOption) (<-chan agentcore.StreamEvent, error) {
	return m.inner.GenerateStream(ctx, messages, tools, stripThinkingOff(opts)...)
}

func (m *summaryModel) SupportsTools() bool { return m.inner.SupportsTools() }

// ProviderName 转发可选接口（agent loop 的 GetApiKey 上下文等用）。
func (m *summaryModel) ProviderName() string {
	if pn, ok := m.inner.(agentcore.ProviderNamer); ok {
		return pn.ProviderName()
	}
	return ""
}

// ModelName 转发可选接口（展示与遥测用）。
func (m *summaryModel) ModelName() string {
	if mn, ok := m.inner.(agentcore.ModelNamer); ok {
		return mn.ModelName()
	}
	return ""
}
