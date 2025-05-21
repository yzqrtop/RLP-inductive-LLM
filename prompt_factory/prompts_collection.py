daiagms_inductive_info_EN = """
1. Extract known conditions or specific cases from the problem.
2. Identify commonalities and summarize patterns (compare multiple cases or conditions involved in the problem horizontally to find commonalities or patterns; if conditions are insufficient, make inferences based on existing experience or commonalities in similar situations)
3. Verify the rationality of the inductive conclusion (verify whether the conclusion drawn conforms to common sense or known rules; if there may be exceptions to the conclusion, further information needs to be supplemented or re inductive)"""

daiagms_reverse_info_EN = """1. Create a new, related but opposite question using the correct answer to the input question.
2. Ensure that the three new answer options are negatively correlated with the three input question options.
3. Ensure that only one answer in the questions you generate is correct and reasonable.
The correct answer in the generated question must appear in the input question.
5. The generated questions and answer choices should be semantically different from the input questions.
"""

daiagms_info_EN = """Inductive thinking mode: a way of thinking from special to general, which extracts general rules or conclusions from specific facts through observation, identification, induction, and application of laws. ‌
Deductive thinking mode: a way of thinking from general to specific, using logical reasoning to derive specific conclusions based on known premises or theories. ‌
Innovative thinking mode: an open and bold way of thinking that encourages independent thinking and the search for novel solutions, typically involving divergent thinking, associative thinking, and creative thinking.
System thinking mode: viewing problems or things as a whole, focusing on the interactions and influences between various elements, and thinking about problems from a comprehensive and global perspective. ‌
Reverse thinking: not following conventional ways of thinking to solve problems, attempting to think from opposite or opposing perspectives, and seeking new solutions. ‌
Critical thinking: the ability to evaluate, analyze, and judge information and viewpoints, approach problems objectively and rationally, and not blindly accept surface information. ‌
Intuitive thinking: making decisions without clear logical reasoning, relying on personal experience, knowledge, and intuitive perception. ‌
Strategic thinking: Thinking about problems from a global perspective, considering long-term and short-term goals, and developing corresponding strategies to achieve optimal results. ‌
Imitation thinking: solving problems by learning and imitating the experiences and practices of others, drawing on their successful experiences and techniques. ‌
Positive thinking: a positive and optimistic way of thinking that focuses on the possibility and positive impact of problem-solving, and faces challenges and difficulties with a positive attitude. ‌
Negative thinking: a negative and pessimistic way of thinking, focusing on the difficulties and obstacles of problems, and easily falling into pessimistic and negative emotions."""

step_by_step_EN = """Let's think step by step. please output the answer"""

daiagms_inductive_info_ZH = """
1. 提取问题中的已知条件或具体案例。
2. 寻找共性，归纳规律（把问题中涉及的多个案例或条件进行横向对比，寻找共性或规律；如果条件不足，基于已有经验或类似情境中的共性进行推测）
3. 验证归纳结论的合理性（验证归纳出的结论是否符合常识或已知规律；如果结论可能存在例外，需要进一步补充信息或重新归纳）"""

daiagms_reverse_info_ZH = """
1. 使用输入问题的正确答案创建一个新的、相关但相反的问题。
2. 确保3个新答案选项与3个输入问题的选项呈负相关。
3. 确保你生成的问题中只有一个答案是正确合理的。
4. 您生成的问题中的正确答案必须出现在输入问题中。
5. 生成的问题和答案选择在语义上应与输入问题不同。"""

daiagms_info_ZH = """归纳思维模式‌：从特殊到一般的思考方式，通过观察、辨认、归纳和应用规律，从具体事实中提炼出一般规律或普遍结论。‌
演绎思维模式‌：从一般到特殊的思考方式，运用逻辑推理，基于已知的前提或理论推导出特定的结论。‌
创新思维模式‌：一种开放、大胆的思考方式，鼓励独立思考和寻找新颖的解决方案，通常涉及发散性思维、联想和创造性思考。
系统思维模式‌：将问题或事物看作一个整体，关注各个元素之间的相互作用和影响，以综合、全局性的视角来思考问题。‌
逆向思维‌：不按常规思维方式来解决问题，尝试从相反或对立的角度进行思考，寻找新的解决方案。‌
批判性思维‌：对信息和观点进行评估、分析和判断的能力，以客观、理性的态度对待问题，不盲目接受表面信息。‌
直觉思维‌：在没有明确逻辑推理的情况下做出决策，依赖于个人的经验、知识和直觉感知。‌
战略思维‌：通过全局的角度来思考问题，考虑长期和短期目标，并制定相应的策略，以实现最佳结果。‌
模仿思维‌：通过学习和模仿他人的经验和做法来解决问题，借鉴他人的成功经验和技巧。‌
正面思维‌：积极、乐观的思维方式，关注解决问题的可能性和积极影响，以积极的心态面对挑战和困难。‌
负面思维‌：消极、悲观的思维方式，关注问题的困难和阻碍，容易陷入悲观和消极的情绪中。"""

step_by_step_ZH = """让我们一步一步地思考。请用中文输出答案。"""



def test_example():
    # we give you an example to test the prompts

    model_id = "your_local_LLM"

    import transformers
    import torch

    pipeline = transformers.pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
    )
    result = {"examples": ["question 1", "question 2", "question 3"]}
    anwer_len = 0
    all_len = 0
    ground_len = 0

    for example in result["examples"]:

        messages = [
            {"role": "system", "content": "You are a knowledge Q&A assistant"},
            {"role": "user",
             "content": f"Q:Question: {example['input']}\n{step_by_step_EN}"},
        ]

        no_daiagms_qa = pipeline(messages, max_new_tokens=256, )[0]["generated_text"][-1]["content"]
        if example["target"] in no_daiagms_qa:
            ground_len += 1

        messages = [
            {"role": "system", "content": "You are a knowledge Q&A assistant"},
            {"role": "user",
             "content": f"Q:Question: {example['input']}\nA:Generate content according to the following process: {daiagms_inductive_info-IN}, answer the final answer"},
        ]

        daiagms_qa = pipeline(messages, max_new_tokens=256, )[0]["generated_text"][-1]["content"]

        daiagms_true_answer = ""
        daiagms_false_answer = ""

        if example["target"] in daiagms_qa:
            anwer_len += 1
            daiagms_true_answer = daiagms_qa
        else:
            daiagms_false_answer = daiagms_qa

        result = {"no_daiagms": no_daiagms_qa, "daiagms": daiagms_qa}