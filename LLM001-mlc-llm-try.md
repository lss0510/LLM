## tvm编译
- 预先安装相关依赖
```bash
sudo apt-get update
sudo apt-get install -y python3 python3-dev python3-setuptools gcc libtinfo-dev zlib1g-dev build-essential cmake libedit-dev libxml2-dev
```
ps：安装完成后需要将python版本升级到3.8，否则后续编译会报错。
还需要将cmake版本升级到3.27
cuda=11.7

- 下载LLVM预编译版本
- 拉取tvm源代码
```bash
git clone --recursive https://github.com/mlc-ai/relax.git tvm-unity
rm -rf build && mkdir build && cd build
cp ../cmake/config.cmake .
```

- 修改config.cmake，set(USE_CUDA ON)，set(USE_LLVM path_of_llvm/bin/llvm-config)，然后make
```bash
vim config.cmake
make -j32
```

- 添加环境变量
```bash
vim ~/.bashrc
// 在文件尾添加
export TVM_HOME=/ssd/ssli/LLM/tvm-unity
export PYTHONPATH=$TVM_HOME/python:$PYTHONPATH
// 保存退出
source ~/.bashrc
```

- 测试

![image.png](https://cdn.nlark.com/yuque/0/2023/png/34282721/1700209187922-f8410c1b-6caf-4524-b244-5a81fb4c394f.png#averageHue=%232d0922&clientId=u5f9add78-2449-4&from=paste&height=271&id=u864a01f0&originHeight=271&originWidth=1275&originalType=binary&ratio=1&rotation=0&showTitle=false&size=61629&status=done&style=none&taskId=u777e5ee8-f6a6-43c3-968a-6db7ef35386&title=&width=1275)
## MLC-LLM编译

- 预先准备-rust安装
   - 配置更新服务器镜像，之后安装
```bash
# 用于更新 toolchain
export RUSTUP_DIST_SERVER=https://mirrors.ustc.edu.cn/rust-static
# 用于更新 rustup
export RUSTUP_UPDATE_ROOT=https://mirrors.ustc.edu.cn/rust-static/rustup

# 安装
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

   - 安装完成

![image.png](https://cdn.nlark.com/yuque/0/2023/png/34282721/1700447827872-2c9a7540-ed48-4a66-b791-78a6c61250f1.png#averageHue=%232d0a23&clientId=u85da4018-51fd-4&from=paste&height=147&id=u746d7b27&originHeight=147&originWidth=552&originalType=binary&ratio=1&rotation=0&showTitle=false&size=22703&status=done&style=none&taskId=u5b143b47-c34a-4607-8c95-fea9b758d96&title=&width=552)

   - 当前shell需要手动激活
```bash
source "$HOME/.cargo/env"
```

   - 为了避免后期build失败，建议创建"~/.cargo/config"文件并写入以下内容更新拉取源
```bash
# 放到 `$HOME/.cargo/config` 文件中
[source.crates-io]
registry = "https://github.com/rust-lang/crates.io-index"

# 替换成你偏好的镜像源
replace-with = 'ustc'

# 清华大学
[source.tuna]
registry = "https://mirrors.tuna.tsinghua.edu.cn/git/crates.io-index.git"

# 中国科学技术大学
[source.ustc]
registry = "git://mirrors.ustc.edu.cn/crates.io-index"

# 上海交通大学
[source.sjtu]
registry = "https://mirrors.sjtug.sjtu.edu.cn/git/crates.io-index"

# rustcc社区
[source.rustcc]
registry = "git://crates.rustcc.cn/crates.io-index"

[net]
git-fetch-with-cli=true
```

- 预先准备-升级gcc版本
   - 安装gcc-8
```bash
sudo apt install gcc-8 gcc-8--multilib g++-8 g++-8--multilib
```

   - 维护gcc系统命令（原本安装的gcc-7）
```bash
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-8 1 --slave /usr/bin/g++ g++ /usr/bin/g++-8
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-7 2 --slave /usr/bin/g++ g++ /usr/bin/g++-7
sudo update-alternatives --config gcc
```

- 拉取MLC-LLM源码，生成cmake_config文件，编译。
```bash
git clone --recursive https://github.com:mlc-ai/mlc-llm.git
cd mlc-llm/cmake
python3 gen_cmake_config.py

cd ..
mkdir build
cp cmake/config.cmake build
cd build
cmake ..
make -j32
```
	由于gcc8不能直接使用#include <filesystem>，这一步会报错。需要在cmakefile中进行修改。
```bash
if(NOT MSVC)
  check_cxx_compiler_flag("-std=c++17" SUPPORT_CXX17)
  set(CMAKE_CXX_FLAGS "-std=c++17 ${CMAKE_CXX_FLAGS}")
  link_libraries(stdc++fs) # 添加上这句
  set(CMAKE_CUDA_STANDARD 17)
else()
  check_cxx_compiler_flag("/std:c++17" SUPPORT_CXX17)
  set(CMAKE_CXX_FLAGS "/std:c++17 ${CMAKE_CXX_FLAGS}")
  set(CMAKE_CUDA_STANDARD 17)
endif()
```

- 测试，输出正常提示信息说明安装成功。
```bash
./mlc_chat_cli -h
```
![image.png](https://cdn.nlark.com/yuque/0/2023/png/34282721/1700469990506-c06297f9-e6a2-41ce-bcb9-d3d601a272fa.png#averageHue=%232e0a23&clientId=u4346f68b-7829-4&from=paste&height=381&id=ubcd8477a&originHeight=381&originWidth=731&originalType=binary&ratio=1&rotation=0&showTitle=false&size=68348&status=done&style=none&taskId=uf7b1b8e6-d42b-4565-bd8d-f88cb698993&title=&width=731)
## 模型量化和运行
下载[RedPajama-INCITE-Chat-3B-v1](https://huggingface.co/togethercomputer/RedPajama-INCITE-Chat-3B-v1/tree/main)大模型，将下载后的模型文件、配置文件存放在mlc-llm/dist/models/RedPajama-INCITE-Chat-3B-v1文件夹下，之后执行量化命令。
```bash
python3 build.py --model MODEL_NAME_OR_PATH --target TARGET_NAME --quantization QUANTIZATION_NAME [--max-seq-len MAX_ALLOWED_SEQUENCE_LENGTH] [--debug-dump] [--use-cache=0]

--model 模型文件夹名称
--target 目标后端平台
--quantization 量化方式

python3 -m mlc_llm.build --hf-path RedPajama-INCITE-Chat-3B-v1 --target cuda --quantization q4f16_1
```
执行完成后会在当前命令所在目录生成dist子目录，存放量化后的模型和配置文件。运行之前build文件夹下生成的mlc_chat_cli，即可启动大模型。
```bash
./build/mlc_chat_cli --model RedPajama-INCITE-Chat-3B-v1
```
![image.png](https://cdn.nlark.com/yuque/0/2023/png/34282721/1700532896374-2d0e45c3-5608-4bc7-bce1-2741adacfb68.png#averageHue=%232d0a23&clientId=u701571c9-0037-4&from=paste&height=314&id=uc3add532&originHeight=314&originWidth=729&originalType=binary&ratio=1&rotation=0&showTitle=false&size=59809&status=done&style=none&taskId=uc3eda9f6-379c-4584-925b-8570168821a&title=&width=729)
