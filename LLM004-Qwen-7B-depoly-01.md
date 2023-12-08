# RedPajama模型生成流程（代码解析版）
RedPajama模型调用gpt_neox.get_model()生成相应模型。PS：只在第一次编译的时候调用get_model函数，之后会直接使用生成好的pkl文件，完成模型的生成。

1. get_model函数输入参数hf_config对应模型文件夹中的config.json文件。根据hf_config生成对应模型config。
2. 注册ParamManager()和relax.Block_Builder()。
3. 根据sep_embed参数判断是否单独调用create_embed_func()函数完成嵌入模块生成。
4. 之后调用create_encoding_func()，create_decoding_func()，create_kv_cache_func()，create_softmax_func()完成计算图的构建。ps：该部分借助同文件下已经编写好的模型框架代码完成。下面给出了一个create_func()函数的详细解读。
```bash
# 函数接受一个relax.BlockBuilder对象bb、一个ParamManager对象param_manager、一个RWKVConfig对象config、
# 一个QuantizationScheme对象quant_scheme和一个字符串类型的函数名称func_name（默认为"prefill"或"decode"）
def create_func(
    bb: relax.BlockBuilder,
    param_manager: ParamManager,
    config: RWKVConfig,
    quant_scheme: QuantizationScheme,
    func_name=Literal["prefill", "decode"],
):
    # 如果函数名称不是"prefill"或"decode"，则抛出ValueError异常。
    if func_name not in ["prefill", "decode"]:
        raise ValueError(f"func_name must be 'prefill' or 'decode', got {func_name}")
    # 根据函数名称确定序列的长度seq_len，如果函数名称为"decode"，则将序列长度设为1，否则设为tir.Var("n", "int64")。
    seq_len = 1 if func_name == "decode" else tir.Var("n", "int64")

    # 在BlockBuilder的function上下文中创建函数func_name。
    with bb.function(func_name):
        # 创建一个RWKVForCausalLM模型对象model。
        model = RWKVForCausalLM(config)
        # 调用param_manager的register_params方法注册模型参数。
        param_manager.register_params(
            model, func_name, quant_scheme, get_param_quant_kind
        )

        # 创建一个输入占位符input_ids，形状为(1, seq_len)，数据类型为"int32"。
        input_ids = nn.Placeholder((1, seq_len), dtype="int32", name="input_ids")
        # Placeholder for compatibility to LLAMA
        # 创建一个占位符all_seq_len_shape，用于兼容LLAMA。
        all_seq_len_shape = relax.Var("place_holder", R.Object())
        # 创建一个变量state，其值为包含多个R.Object()的元组，长度为config.num_hidden_layers * 5。
        state = relax.Var("state", R.Tuple([R.Object()] * config.num_hidden_layers * 5))
        with bb.dataflow():
            # 调用model的前向传播函数，将input_ids和state作为输入，得到输出logits和状态列表states。
            logits, states = model(input_ids, state)
            # 将input_ids、all_seq_len_shape、state和模型的参数列表作为参数列表params。
            params = [
                input_ids,
                all_seq_len_shape,
                state,
            ] + model.parameters()

            # 使用bb.emit_output将(logits, relax.Tuple(states))作为输出。
            gv = bb.emit_output((logits, relax.Tuple(states)))
        # 使用bb.emit_func_output将输出和参数列表params作为函数的输出。
        bb.emit_func_output(gv, params)

    # 获取构建好的模块mod和global function变量gv。
    mod = bb.get()
    gv = mod.get_global_var(func_name)
    # 根据函数名称更新函数的属性，包括输入数量和tir_var_upper_bound（如果函数名称为"prefill"）。
    f = mod[gv].with_attr("num_input", 3)
    if func_name == "prefill":
        f = f.with_attr("tir_var_upper_bound", {"n": config.max_sequence_length})
    bb.update_func(gv, f)
```

5. 分别调用f_convert_pname_fwd()，f_convert_param_bkwd()完成前向参数、反向参数的转换。最后调用param_manager的set_param_loading_func方法，设置参数加载函数。

_参考资料：_[_mlc-llm 推理优化和大语言模型搭建解析_](https://zhuanlan.zhihu.com/p/658354795)
# Qwen模型搭建、生成：
Qwen模型和Llama模型结构基本相同，唯一的区别在于QwenAttention模块中c_attn线性层有偏置参数，注意这一点，再对Llama模块中各部分的参数名进行修改，即可完成Qwen模型的搭建。之后仿照上面的模型生成流程，编写相应的create_func函数，即可完成Qwen模型的生成。
# Qwen模型服务器cuda编译、部署
为了测试上面生成的模型是否能正确的编译，尝试在cuda下进行部署。编译过程很顺利，但是在部署时出现错误。可以看到模型的输出乱码，debug发现是分词器匹配不当导致的问题。
![image.png](https://cdn.nlark.com/yuque/0/2023/png/34282721/1702002988920-b8d8830a-21b4-47bf-a8d6-4de9824a7bdb.png#averageHue=%232d0923&clientId=u805301c8-668b-4&from=paste&height=310&id=ude9f28e1&originHeight=310&originWidth=1629&originalType=binary&ratio=1&rotation=0&showTitle=false&size=68601&status=done&style=none&taskId=u32b15365-5a9c-437a-80dd-9493cc2d107&title=&width=1629)
# NLP中**tokenzier的作用**
tokenzier的主要功能是将文本编码为模型可以理解的形式，以及将模型的输出解码为人类可以理解的文本。在自然语言处理（NLP）任务中，分词是非常重要的一步。原始的文本数据不能直接被大多数的机器学习模型处理，因为这些模型需要数值输入。分词器可以将文本数据转换为数值形式，使得模型可以处理。具体来说，分词器将文本分割成更小的单元（如词、子词或字符），然后将这些单元转换为对应的数值（如整数或浮点数）。这些数值可以被模型用来学习文本的语义和结构。此外，分词器还可以处理一些特殊的标记，如开始标记（BOS）、结束标记（EOS）和填充标记（PAD）。这些标记在训练和生成序列时非常重要。总的来说， Tokenizer类是一个非常重要的工具，它可以帮助你处理文本数据，使得你的模型可以理解和生成文本。
# HuggingFace AutoTokenizer.from_pretrained()流程分析
transformers/models/auto/tokenization_auto.py

1. 调用get_tokenizer_config()，get_tokenizer_config()再调用cached_file()读取tokenizer_config.json文件。![image.png](https://cdn.nlark.com/yuque/0/2023/png/34282721/1701845490053-e5045453-a504-4071-b0c7-8a5bae298cdb.png#averageHue=%2322211f&clientId=u61cbc424-6493-4&from=paste&height=43&id=ua06d71d5&originHeight=43&originWidth=943&originalType=binary&ratio=1&rotation=0&showTitle=false&size=16214&status=done&style=none&taskId=u66d7d1ce-c4fb-445c-a971-029955ae3d9&title=&width=943)![image.png](https://cdn.nlark.com/yuque/0/2023/png/34282721/1701845885719-f8473456-5b26-4888-8c6c-d8580c0ef28d.png#averageHue=%231f1f1e&clientId=u61cbc424-6493-4&from=paste&height=304&id=u2ce845fe&originHeight=304&originWidth=716&originalType=binary&ratio=1&rotation=0&showTitle=false&size=58225&status=done&style=none&taskId=u23938e63-8dd6-496f-8a46-b4a6a503e30&title=&width=716)![image.png](https://cdn.nlark.com/yuque/0/2023/png/34282721/1701845912623-2ad716ce-fdf2-49d1-b61d-ac24a13f4291.png#averageHue=%23fbfbfb&clientId=u61cbc424-6493-4&from=paste&height=187&id=uc9f44b38&originHeight=187&originWidth=400&originalType=binary&ratio=1&rotation=0&showTitle=false&size=9464&status=done&style=none&taskId=ufe983888-c8a6-47fe-a461-415a3c19607&title=&width=400)
2. 根据tokenizer_config.json中的tokenizer_class得到config_tokenizer_class为QwenTokenizer。![image.png](https://cdn.nlark.com/yuque/0/2023/png/34282721/1701846114791-dfe2be39-dfa6-4ede-9206-3eccda3b1cb0.png#averageHue=%23242220&clientId=u61cbc424-6493-4&from=paste&height=24&id=u34526e0b&originHeight=24&originWidth=801&originalType=binary&ratio=1&rotation=0&showTitle=false&size=8225&status=done&style=none&taskId=uf6b3a490-d6a0-461c-ae8c-16ad15f650a&title=&width=801)
3. 由于配置文件中含有auto_map关键词，读取到tokenizer_auto_map=["tokenization_qwen.QWenTokenizer", None]。![image.png](https://cdn.nlark.com/yuque/0/2023/png/34282721/1701846610466-0d9283ab-f844-40fd-98fd-d463b5113395.png#averageHue=%2322201f&clientId=u61cbc424-6493-4&from=paste&height=130&id=CijqI&originHeight=130&originWidth=918&originalType=binary&ratio=1&rotation=0&showTitle=false&size=34878&status=done&style=none&taskId=ub36de171-dbe4-489a-b502-8f47dc15a7d&title=&width=918)
4. 根据has_remote_code=True、trust_remote_code=True，调用get_class_from_dynamic_module()。![image.png](https://cdn.nlark.com/yuque/0/2023/png/34282721/1701846954268-b129c92c-e809-4912-8068-388d96854992.png#averageHue=%23201f1e&clientId=u61cbc424-6493-4&from=paste&height=321&id=ue707b463&originHeight=321&originWidth=1065&originalType=binary&ratio=1&rotation=0&showTitle=false&size=90273&status=done&style=none&taskId=u960f2d5e-69f1-4cfd-8b1f-552794b05f4&title=&width=1065)
5. 分离出module_file=tokenization_qwen，class_name=QWenTokenizer，调用get_cached_module_file()，之后调用get_class_in_module()。![image.png](https://cdn.nlark.com/yuque/0/2023/png/34282721/1701848782420-4d842e5d-08eb-4e5b-a944-d3bc8d8c47c1.png#averageHue=%231f1f1e&clientId=u61cbc424-6493-4&from=paste&height=382&id=u62452381&originHeight=382&originWidth=835&originalType=binary&ratio=1&rotation=0&showTitle=false&size=76934&status=done&style=none&taskId=u4e77409c-8506-4560-ae7f-ee235d61277&title=&width=835)![image.png](https://cdn.nlark.com/yuque/0/2023/png/34282721/1701849022105-b256b939-e1e4-4637-8258-05d0613807ae.png#averageHue=%2322201f&clientId=u61cbc424-6493-4&from=paste&height=293&id=u3e5d7e05&originHeight=293&originWidth=930&originalType=binary&ratio=1&rotation=0&showTitle=false&size=60373&status=done&style=none&taskId=ub4ae0209-06f2-4e9d-930e-c926b8f0272&title=&width=930)
6. 上一步将参数返回给了tokenizer_class， 最后from_pretrained。下面可![image.png](https://cdn.nlark.com/yuque/0/2023/png/34282721/1701849251788-42ac5cf3-3947-4c7d-bba3-8a8a68b9011a.png#averageHue=%232c1f1e&clientId=u61cbc424-6493-4&from=paste&height=205&id=ud142b6d5&originHeight=205&originWidth=1103&originalType=binary&ratio=1&rotation=0&showTitle=false&size=60250&status=done&style=none&taskId=u638019b6-3879-46b2-8cf8-23d2c39beda&title=&width=1103)以看tokenization_qwen.py中相应的代码。tokenization_qwen.py中QWenTokenizer继承自PreTrainedTokenizer，PreTrainedTokenizer继承自PreTrainedTokenizerBase。
7. PreTrainedTokenizerBase中有from_pretrained()，该函数搜索相应的文件。之后调用_from_pretrained()，最后貌似就是将参数梳理了一遍，再调用类注册一个tokenizer。![image.png](https://cdn.nlark.com/yuque/0/2023/png/34282721/1701850741269-c3aefffb-c9dc-420e-8115-10c0f3bd2214.png#averageHue=%23201f1e&clientId=u61cbc424-6493-4&from=paste&height=542&id=ufa43df37&originHeight=542&originWidth=932&originalType=binary&ratio=1&rotation=0&showTitle=false&size=120316&status=done&style=none&taskId=u5f6611d4-d413-4c18-ab19-cb4dfc50384&title=&width=932)![image.png](https://cdn.nlark.com/yuque/0/2023/png/34282721/1701850788196-fbf9198e-66e8-4b3b-bf12-6b09cef0b726.png#averageHue=%232e0a25&clientId=u61cbc424-6493-4&from=paste&height=58&id=u98e0aef7&originHeight=58&originWidth=732&originalType=binary&ratio=1&rotation=0&showTitle=false&size=15664&status=done&style=none&taskId=u57c29ce2-5f18-44fe-a163-383b375db54&title=&width=732)![image.png](https://cdn.nlark.com/yuque/0/2023/png/34282721/1701850997654-0f4552b1-de52-4742-9b29-65fd5c9e7864.png#averageHue=%23211f1e&clientId=u61cbc424-6493-4&from=paste&height=221&id=u85baa1fe&originHeight=221&originWidth=549&originalType=binary&ratio=1&rotation=0&showTitle=false&size=37508&status=done&style=none&taskId=udd07cf6f-64bc-4375-a210-a8af473b04f&title=&width=549)![image.png](https://cdn.nlark.com/yuque/0/2023/png/34282721/1701911513009-964cb1cc-c8dc-4b9f-a929-911244c3c759.png#averageHue=%2322201f&clientId=u61cbc424-6493-4&from=paste&height=280&id=uf1bfdd19&originHeight=280&originWidth=1081&originalType=binary&ratio=1&rotation=0&showTitle=false&size=63817&status=done&style=none&taskId=u44673ea3-9566-4918-b527-1e468dc857a&title=&width=1081)
# Tokenization_qwen编解码流程
```python
# 注册qwen_tokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-1_8B-Chat", trust_remote_code=True)

# 分词
word_segmentation_result = tokenizer.tokenize("你好,你是谁？")
print("\"你好,你是谁？\"分词结果:", word_segmentation_result)

# 编码
encode_result = tokenizer.convert_tokens_to_ids(word_segmentation_result)
print("\"你好,你是谁？\"编码结果:", encode_result)

# 直接编码
input_ids = tokenizer.encode("你好,你是谁？")
print("\"你好,你是谁？\"直接编码结果:", input_ids)

# 返回分词结果
word_segmentation_result_d = []
for ids in input_ids:
    token = tokenizer._convert_id_to_token(ids)
    word_segmentation_result_d.append(token)
print("\"你好,你是谁？\"编码返回分词结果:", word_segmentation_result_d)

# 返回string
string_in = tokenizer.convert_tokens_to_string(word_segmentation_result_d)
print("\"你好,你是谁？\"分词返回输入结果:", string_in)
```
![image.png](https://cdn.nlark.com/yuque/0/2023/png/34282721/1701960625771-5631f8f6-6b0e-45f3-9ceb-86a71dde95a8.png#averageHue=%23391830&clientId=u5c098190-7cf5-4&from=paste&height=72&id=u0185044b&originHeight=108&originWidth=1182&originalType=binary&ratio=1.5&rotation=0&showTitle=false&size=133746&status=done&style=none&taskId=u8c4ac171-6a05-4b49-9437-9f899f8c87f&title=&width=788)
其中由分词结果到编码结果实际就是查询vacab的过程，主要的问题在于如果由字符串变为tokens。这一部分对应的代码如下：
![image.png](https://cdn.nlark.com/yuque/0/2023/png/34282721/1701961165353-fe564154-f72f-42f4-95ed-32a2dcfaa0ed.png#averageHue=%2325201f&clientId=u5c098190-7cf5-4&from=paste&height=470&id=ua6911e19&originHeight=705&originWidth=868&originalType=binary&ratio=1.5&rotation=0&showTitle=false&size=265906&status=done&style=none&taskId=u4fe16ece-44e5-44b6-bfef-f02862d6895&title=&width=578.6666666666666)
从上图可以看出这一部分调用的是self.tokenizer.encode()，该函数如下，其主要调用的是self._core_bpe.encode()，self._core_bpe则是由已经编译好的python库注册而来。![image.png](https://cdn.nlark.com/yuque/0/2023/png/34282721/1701961338603-ccaca60c-ee9f-43a4-a6e3-a956f9f7633c.png#averageHue=%23232020&clientId=u5c098190-7cf5-4&from=paste&height=736&id=uf6fe87dd&originHeight=1104&originWidth=1162&originalType=binary&ratio=1.5&rotation=0&showTitle=false&size=624817&status=done&style=none&taskId=ueb5d0c5a-8970-4f93-934a-617b8654048&title=&width=774.6666666666666)![image.png](https://cdn.nlark.com/yuque/0/2023/png/34282721/1701961513917-580dab15-5828-4d66-9d6f-1d953924ae32.png#averageHue=%2337342a&clientId=u5c098190-7cf5-4&from=paste&height=18&id=ud73d4158&originHeight=27&originWidth=706&originalType=binary&ratio=1.5&rotation=0&showTitle=false&size=20221&status=done&style=none&taskId=uf4aab8f9-aa8f-4bac-842b-c1bb8b5e440&title=&width=470.6666666666667)

