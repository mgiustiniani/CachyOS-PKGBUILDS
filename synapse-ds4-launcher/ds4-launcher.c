#define _XOPEN_SOURCE 700
#define _POSIX_C_SOURCE 200809L

#include <ctype.h>
#include <errno.h>
#include <limits.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#ifndef PATH_MAX
#define PATH_MAX 4096
#endif

#define MAX_PROFILES 48
#define MAX_VARIABLES 32
#define MAX_ARGUMENTS 80
#define MAX_ENVIRONMENT 24
#define MAX_EXECUTABLES 8
#define MAX_SCOPED_ARGUMENTS 80
#define MAX_OVERRIDES 32
#define MAX_EXTRA_ARGS 64
#define MAX_LINE 4096

struct variable {
    char *name;
    char *label;
    char *default_value;
    char *value;
    char *executable;
    bool supplied;
};

struct scoped_argument {
    char *executable;
    char *value;
};

struct profile {
    char *id;
    char *name;
    char *description;
    char *ds4_dir;
    char *workdir;
    char *executables[MAX_EXECUTABLES];
    size_t executable_count;
    struct variable variables[MAX_VARIABLES];
    size_t variable_count;
    char *arguments[MAX_ARGUMENTS];
    size_t argument_count;
    struct scoped_argument scoped_arguments[MAX_SCOPED_ARGUMENTS];
    size_t scoped_argument_count;
    char *environment[MAX_ENVIRONMENT];
    size_t environment_count;
};

struct override {
    char *name;
    char *value;
};

struct options {
    const char *ds4_dir;
    const char *workdir;
    const char *profile_id;
    const char *executable;
    const char *profiles_file;
    bool list_profiles;
    bool list_executables;
    bool dry_run;
    bool no_prompt;
    bool assume_yes;
    bool show_help;
    struct override overrides[MAX_OVERRIDES];
    size_t override_count;
    char *extra_args[MAX_EXTRA_ARGS];
    size_t extra_arg_count;
};

static struct profile profiles[MAX_PROFILES];
static size_t profile_count;

static char *expand_value(const char *input, const struct profile *profile,
                          const char *executable);

static const char *known_executables[] = {
    "ds4", "ds4-server", "ds4-bench", "ds4-eval", "ds4-agent"
};

static const char builtin_profiles[] =
"# Built-in profiles derived from the DS4 README and validated two-node runbook.\n"
"[profile local-chat]\n"
"name=Local interactive chat\n"
"executable=ds4\n"
"description=Open an interactive DS4 chat with a local model.\n"
"variable=MODEL|GGUF model|gguf/DeepSeek-V4-Flash-Q4KExperts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out-chat-v2-imatrix.gguf\n"
"variable=CTX|Allocated context|100000\n"
"argument=-m\n"
"argument=${MODEL}\n"
"argument=-c\n"
"argument=${CTX}\n"
"\n"
"[profile local-one-shot]\n"
"name=Local one-shot prompt\n"
"executable=ds4\n"
"description=Run one prompt and exit.\n"
"variable=MODEL|GGUF model|gguf/DeepSeek-V4-Flash-Q4KExperts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out-chat-v2-imatrix.gguf\n"
"variable=CTX|Allocated context|32768\n"
"variable=PROMPT|Prompt|Explain the current system status concisely.\n"
"variable=TOKENS|Maximum generated tokens|1024\n"
"argument=-m\n"
"argument=${MODEL}\n"
"argument=-c\n"
"argument=${CTX}\n"
"argument=-n\n"
"argument=${TOKENS}\n"
"argument=-p\n"
"argument=${PROMPT}\n"
"\n"
"[profile glm52-dgx-strix]\n"
"name=GLM 5.2 DGX Spark + Strix Halo coordinator\n"
"executables=ds4,ds4-server,ds4-bench,ds4-eval,ds4-agent\n"
"description=Validated DGX Spark 0:41 coordinator configuration with Strix 42:output worker.\n"
"variable=MODEL|GGUF model|gguf/GLM-5.2-UD-IQ2_XXS_RoutedIQ2XXS_blk78Q2K.gguf\n"
"variable=LISTEN_IP|Coordinator listen IP|10.44.0.1\n"
"variable=DIST_PORT|Distributed control port|19800\n"
"variable=CTX|Allocated context|2304\n"
"variable=CUDA_Q8_F16_CACHE_MB|CUDA Q8-to-F16 cache MiB|7168\n"
"variable.ds4-server=HOST|HTTP bind IP|127.0.0.1\n"
"variable.ds4-server=HTTP_PORT|HTTP port|8000\n"
"variable.ds4-server=KV_DIR|Disk KV directory|${HOME}/.ds4/glm52-server-kv\n"
"variable.ds4-bench=PROMPT_FILE|Benchmark prompt file|speed-bench/promessi_sposi.txt\n"
"variable.ds4-bench=CSV|CSV output|${HOME}/glm52-distributed-2048.csv\n"
"variable.ds4-eval=TRACE|Evaluation trace|${HOME}/glm52-eval.trace\n"
"variable.ds4-agent=CHDIR|Agent working directory|${HOME}/workspace\n"
"environment=DS4_GLM_MEMORY_GUARD=0\n"
"environment=DS4_CUDA_WEIGHT_CACHE_LIMIT_GB=108\n"
"environment=DS4_CUDA_WEIGHT_ARENA_CHUNK_MB=256\n"
"environment=DS4_CUDA_Q8_F16_CACHE_MB=${CUDA_Q8_F16_CACHE_MB}\n"
"environment=DS4_CUDA_GLM_COMPACT_CACHE_F16=1\n"
"argument=-m\n"
"argument=${MODEL}\n"
"argument=--role\n"
"argument=coordinator\n"
"argument=--layers\n"
"argument=0:41\n"
"argument=--listen\n"
"argument=${LISTEN_IP}\n"
"argument=${DIST_PORT}\n"
"argument.ds4=-c\n"
"argument.ds4=${CTX}\n"
"argument.ds4-server=-c\n"
"argument.ds4-server=${CTX}\n"
"argument.ds4-server=--host\n"
"argument.ds4-server=${HOST}\n"
"argument.ds4-server=--port\n"
"argument.ds4-server=${HTTP_PORT}\n"
"argument.ds4-server=--kv-disk-dir\n"
"argument.ds4-server=${KV_DIR}\n"
"argument.ds4-server=--kv-disk-space-mb\n"
"argument.ds4-server=8192\n"
"argument.ds4-bench=--prompt-file\n"
"argument.ds4-bench=${PROMPT_FILE}\n"
"argument.ds4-bench=--ctx-start\n"
"argument.ds4-bench=2048\n"
"argument.ds4-bench=--ctx-max\n"
"argument.ds4-bench=2048\n"
"argument.ds4-bench=--ctx-alloc\n"
"argument.ds4-bench=2177\n"
"argument.ds4-bench=--gen-tokens\n"
"argument.ds4-bench=128\n"
"argument.ds4-bench=--dist-prefill-chunk\n"
"argument.ds4-bench=512\n"
"argument.ds4-bench=--dist-prefill-window\n"
"argument.ds4-bench=2\n"
"argument.ds4-bench=--debug\n"
"argument.ds4-bench=--csv\n"
"argument.ds4-bench=${CSV}\n"
"argument.ds4-eval=--plain\n"
"argument.ds4-eval=--questions\n"
"argument.ds4-eval=1\n"
"argument.ds4-eval=--tokens\n"
"argument.ds4-eval=128\n"
"argument.ds4-eval=--trace\n"
"argument.ds4-eval=${TRACE}\n"
"argument.ds4-agent=-c\n"
"argument.ds4-agent=${CTX}\n"
"argument.ds4-agent=--chdir\n"
"argument.ds4-agent=${CHDIR}\n"
"\n"
"[profile glm-worker-strix]\n"
"name=GLM 5.2 worker on Strix Halo\n"
"executable=ds4\n"
"description=Validated worker split 42:output for the DGX Spark coordinator.\n"
"variable=MODEL|GGUF model|gguf/GLM-5.2-UD-IQ2_XXS_RoutedIQ2XXS_blk78Q2K.gguf\n"
"variable=COORDINATOR_IP|Coordinator IP|10.44.0.1\n"
"variable=PORT|Distributed control port|19800\n"
"variable=CTX|Allocated context|2304\n"
"environment=DS4_GLM_MEMORY_GUARD=0\n"
"environment=DS4_ROCM_GLM_COMPACT_CACHE_F16=1\n"
"argument=-m\n"
"argument=${MODEL}\n"
"argument=-c\n"
"argument=${CTX}\n"
"argument=--role\n"
"argument=worker\n"
"argument=--layers\n"
"argument=42:output\n"
"argument=--coordinator\n"
"argument=${COORDINATOR_IP}\n"
"argument=${PORT}\n"
"\n"
"[profile q4-worker-strix]\n"
"name=DeepSeek V4 Flash Q4 worker on Strix Halo\n"
"executable=ds4\n"
"description=Validated worker split 22:output with context for the full sweep.\n"
"variable=MODEL|GGUF model|gguf/DeepSeek-V4-Flash-Q4KExperts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out-chat-v2-imatrix.gguf\n"
"variable=COORDINATOR_IP|Coordinator IP|10.44.0.1\n"
"variable=PORT|Distributed control port|19800\n"
"variable=CTX|Allocated context|65665\n"
"argument=-m\n"
"argument=${MODEL}\n"
"argument=-c\n"
"argument=${CTX}\n"
"argument=--role\n"
"argument=worker\n"
"argument=--layers\n"
"argument=22:output\n"
"argument=--coordinator\n"
"argument=${COORDINATOR_IP}\n"
"argument=${PORT}\n"
"\n"
"[profile q4-distributed-chat-spark]\n"
"name=DeepSeek V4 Flash Q4 distributed chat on DGX Spark\n"
"executable=ds4\n"
"description=Coordinator split 0:21 for interactive chat with a Strix worker.\n"
"variable=MODEL|GGUF model|gguf/DeepSeek-V4-Flash-Q4KExperts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out-chat-v2-imatrix.gguf\n"
"variable=LISTEN_IP|Coordinator listen IP|10.44.0.1\n"
"variable=PORT|Distributed control port|19800\n"
"variable=CTX|Allocated context|65665\n"
"environment=DS4_GLM_MEMORY_GUARD=0\n"
"environment=DS4_CUDA_WEIGHT_CACHE_LIMIT_GB=116\n"
"environment=DS4_CUDA_WEIGHT_ARENA_CHUNK_MB=256\n"
"argument=-m\n"
"argument=${MODEL}\n"
"argument=-c\n"
"argument=${CTX}\n"
"argument=--role\n"
"argument=coordinator\n"
"argument=--layers\n"
"argument=0:21\n"
"argument=--listen\n"
"argument=${LISTEN_IP}\n"
"argument=${PORT}\n"
"\n"
"[profile local-server]\n"
"name=Local OpenAI-compatible server\n"
"executable=ds4-server\n"
"description=Serve a local model with persistent disk KV cache.\n"
"variable=MODEL|GGUF model|gguf/DeepSeek-V4-Flash-Q4KExperts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out-chat-v2-imatrix.gguf\n"
"variable=HOST|HTTP bind IP|127.0.0.1\n"
"variable=HTTP_PORT|HTTP port|8000\n"
"variable=CTX|Allocated context|100000\n"
"variable=KV_DIR|Disk KV directory|${HOME}/.ds4/server-kv\n"
"variable=KV_MB|Disk KV budget in MiB|8192\n"
"argument=-m\n"
"argument=${MODEL}\n"
"argument=-c\n"
"argument=${CTX}\n"
"argument=--host\n"
"argument=${HOST}\n"
"argument=--port\n"
"argument=${HTTP_PORT}\n"
"argument=--kv-disk-dir\n"
"argument=${KV_DIR}\n"
"argument=--kv-disk-space-mb\n"
"argument=${KV_MB}\n"
"\n"
"[profile q4-distributed-server-spark]\n"
"name=Distributed Q4 server on DGX Spark\n"
"executable=ds4-server\n"
"description=OpenAI-compatible coordinator using the validated Q4 split.\n"
"variable=MODEL|GGUF model|gguf/DeepSeek-V4-Flash-Q4KExperts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out-chat-v2-imatrix.gguf\n"
"variable=LISTEN_IP|Distributed listen IP|10.44.0.1\n"
"variable=DIST_PORT|Distributed control port|19800\n"
"variable=HOST|HTTP bind IP|127.0.0.1\n"
"variable=HTTP_PORT|HTTP port|8000\n"
"variable=CTX|Allocated context|65665\n"
"variable=KV_DIR|Disk KV directory|${HOME}/.ds4/server-kv\n"
"environment=DS4_GLM_MEMORY_GUARD=0\n"
"environment=DS4_CUDA_WEIGHT_CACHE_LIMIT_GB=116\n"
"environment=DS4_CUDA_WEIGHT_ARENA_CHUNK_MB=256\n"
"argument=-m\n"
"argument=${MODEL}\n"
"argument=-c\n"
"argument=${CTX}\n"
"argument=--host\n"
"argument=${HOST}\n"
"argument=--port\n"
"argument=${HTTP_PORT}\n"
"argument=--kv-disk-dir\n"
"argument=${KV_DIR}\n"
"argument=--kv-disk-space-mb\n"
"argument=8192\n"
"argument=--role\n"
"argument=coordinator\n"
"argument=--layers\n"
"argument=0:21\n"
"argument=--listen\n"
"argument=${LISTEN_IP}\n"
"argument=${DIST_PORT}\n"
"\n"
"[profile local-benchmark]\n"
"name=Local context benchmark\n"
"executable=ds4-bench\n"
"description=Canonical local benchmark from 2048 to 65536 tokens.\n"
"variable=MODEL|GGUF model|ds4flash.gguf\n"
"variable=PROMPT_FILE|Benchmark prompt file|speed-bench/promessi_sposi.txt\n"
"variable=CTX_START|First context frontier|2048\n"
"variable=CTX_MAX|Last context frontier|65536\n"
"variable=STEP|Context increment|2048\n"
"variable=GEN_TOKENS|Generated tokens per frontier|128\n"
"variable=CSV|CSV output|${HOME}/ds4-local-bench.csv\n"
"argument=-m\n"
"argument=${MODEL}\n"
"argument=--prompt-file\n"
"argument=${PROMPT_FILE}\n"
"argument=--ctx-start\n"
"argument=${CTX_START}\n"
"argument=--ctx-max\n"
"argument=${CTX_MAX}\n"
"argument=--step-incr\n"
"argument=${STEP}\n"
"argument=--gen-tokens\n"
"argument=${GEN_TOKENS}\n"
"argument=--csv\n"
"argument=${CSV}\n"
"\n"
"[profile glm-benchmark-spark]\n"
"name=GLM 5.2 distributed 2048/128 benchmark on DGX Spark\n"
"executable=ds4-bench\n"
"description=Validated GLM coordinator split 0:41 with Strix worker 42:output.\n"
"variable=MODEL|GGUF model|gguf/GLM-5.2-UD-IQ2_XXS_RoutedIQ2XXS_blk78Q2K.gguf\n"
"variable=PROMPT_FILE|Benchmark prompt file|speed-bench/promessi_sposi.txt\n"
"variable=LISTEN_IP|Coordinator listen IP|10.44.0.1\n"
"variable=PORT|Distributed control port|19800\n"
"variable=CSV|CSV output|${HOME}/glm52-distributed-2048.csv\n"
"environment=DS4_GLM_MEMORY_GUARD=0\n"
"environment=DS4_CUDA_WEIGHT_CACHE_LIMIT_GB=108\n"
"environment=DS4_CUDA_WEIGHT_ARENA_CHUNK_MB=256\n"
"environment=DS4_CUDA_Q8_F16_CACHE_MB=7168\n"
"environment=DS4_CUDA_GLM_COMPACT_CACHE_F16=1\n"
"argument=-m\n"
"argument=${MODEL}\n"
"argument=--prompt-file\n"
"argument=${PROMPT_FILE}\n"
"argument=--ctx-start\n"
"argument=2048\n"
"argument=--ctx-max\n"
"argument=2048\n"
"argument=--ctx-alloc\n"
"argument=2177\n"
"argument=--gen-tokens\n"
"argument=128\n"
"argument=--dist-prefill-chunk\n"
"argument=512\n"
"argument=--dist-prefill-window\n"
"argument=2\n"
"argument=--role\n"
"argument=coordinator\n"
"argument=--layers\n"
"argument=0:41\n"
"argument=--listen\n"
"argument=${LISTEN_IP}\n"
"argument=${PORT}\n"
"argument=--debug\n"
"argument=--csv\n"
"argument=${CSV}\n"
"\n"
"[profile q4-benchmark-spark]\n"
"name=Q4 distributed 2048/128 benchmark on DGX Spark\n"
"executable=ds4-bench\n"
"description=Validated Q4 coordinator split 0:21 with Strix worker 22:output.\n"
"variable=MODEL|GGUF model|gguf/DeepSeek-V4-Flash-Q4KExperts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out-chat-v2-imatrix.gguf\n"
"variable=PROMPT_FILE|Benchmark prompt file|speed-bench/promessi_sposi.txt\n"
"variable=LISTEN_IP|Coordinator listen IP|10.44.0.1\n"
"variable=PORT|Distributed control port|19800\n"
"variable=CSV|CSV output|${HOME}/q4k-tile16-2048.csv\n"
"environment=DS4_GLM_MEMORY_GUARD=0\n"
"environment=DS4_CUDA_WEIGHT_CACHE_LIMIT_GB=116\n"
"environment=DS4_CUDA_WEIGHT_ARENA_CHUNK_MB=256\n"
"argument=-m\n"
"argument=${MODEL}\n"
"argument=--prompt-file\n"
"argument=${PROMPT_FILE}\n"
"argument=--ctx-start\n"
"argument=2048\n"
"argument=--ctx-max\n"
"argument=2048\n"
"argument=--ctx-alloc\n"
"argument=2177\n"
"argument=--gen-tokens\n"
"argument=128\n"
"argument=--dist-prefill-chunk\n"
"argument=512\n"
"argument=--dist-prefill-window\n"
"argument=2\n"
"argument=--role\n"
"argument=coordinator\n"
"argument=--layers\n"
"argument=0:21\n"
"argument=--listen\n"
"argument=${LISTEN_IP}\n"
"argument=${PORT}\n"
"argument=--debug\n"
"argument=--csv\n"
"argument=${CSV}\n"
"\n"
"[profile q4-full-sweep-spark]\n"
"name=Q4 distributed full 2048-65536 sweep on DGX Spark\n"
"executable=ds4-bench\n"
"description=Validated 32-frontier Q4 sweep with persistent CSV output.\n"
"variable=MODEL|GGUF model|gguf/DeepSeek-V4-Flash-Q4KExperts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out-chat-v2-imatrix.gguf\n"
"variable=PROMPT_FILE|Benchmark prompt file|speed-bench/promessi_sposi.txt\n"
"variable=LISTEN_IP|Coordinator listen IP|10.44.0.1\n"
"variable=PORT|Distributed control port|19800\n"
"variable=CSV|CSV output|${HOME}/q4k-tile16-full-sweep.csv\n"
"environment=DS4_GLM_MEMORY_GUARD=0\n"
"environment=DS4_CUDA_WEIGHT_CACHE_LIMIT_GB=116\n"
"environment=DS4_CUDA_WEIGHT_ARENA_CHUNK_MB=256\n"
"argument=-m\n"
"argument=${MODEL}\n"
"argument=--prompt-file\n"
"argument=${PROMPT_FILE}\n"
"argument=--ctx-start\n"
"argument=2048\n"
"argument=--ctx-max\n"
"argument=65536\n"
"argument=--step-incr\n"
"argument=2048\n"
"argument=--ctx-alloc\n"
"argument=65665\n"
"argument=--gen-tokens\n"
"argument=128\n"
"argument=--dist-prefill-chunk\n"
"argument=512\n"
"argument=--dist-prefill-window\n"
"argument=2\n"
"argument=--role\n"
"argument=coordinator\n"
"argument=--layers\n"
"argument=0:21\n"
"argument=--listen\n"
"argument=${LISTEN_IP}\n"
"argument=${PORT}\n"
"argument=--debug\n"
"argument=--csv\n"
"argument=${CSV}\n"
"\n"
"[profile deterministic-eval]\n"
"name=Deterministic four-question evaluation\n"
"executable=ds4-eval\n"
"description=The documented q1-q4 capability regression gate.\n"
"variable=MODEL|GGUF model|ds4flash.gguf\n"
"variable=TRACE|Trace output|${HOME}/ds4-eval.trace\n"
"argument=-m\n"
"argument=${MODEL}\n"
"argument=--plain\n"
"argument=--questions\n"
"argument=4\n"
"argument=--tokens\n"
"argument=2048\n"
"argument=--temp\n"
"argument=0\n"
"argument=--seed\n"
"argument=1\n"
"argument=--trace\n"
"argument=${TRACE}\n"
"\n"
"[profile local-agent]\n"
"name=Local native coding agent\n"
"executable=ds4-agent\n"
"description=Start the interactive native agent from the DS4 project directory.\n"
"variable=MODEL|GGUF model|gguf/DeepSeek-V4-Flash-Q4KExperts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out-chat-v2-imatrix.gguf\n"
"variable=CTX|Allocated context|100000\n"
"variable=CHDIR|Agent working directory|${HOME}/workspace\n"
"argument=-m\n"
"argument=${MODEL}\n"
"argument=-c\n"
"argument=${CTX}\n"
"argument=--chdir\n"
"argument=${CHDIR}\n";

static void die(const char *message)
{
    fprintf(stderr, "ds4-launcher: %s\n", message);
    exit(EXIT_FAILURE);
}

static void die_errno(const char *context)
{
    fprintf(stderr, "ds4-launcher: %s: %s\n", context, strerror(errno));
    exit(EXIT_FAILURE);
}

static char *xstrdup(const char *value)
{
    char *copy = strdup(value ? value : "");
    if (!copy)
        die("out of memory");
    return copy;
}

static char *trim(char *value)
{
    char *end;
    while (isspace((unsigned char)*value))
        value++;
    if (*value == '\0')
        return value;
    end = value + strlen(value) - 1;
    while (end > value && isspace((unsigned char)*end))
        *end-- = '\0';
    return value;
}

static struct profile *find_profile(const char *id)
{
    size_t i;
    for (i = 0; i < profile_count; i++) {
        if (profiles[i].id && strcmp(profiles[i].id, id) == 0)
            return &profiles[i];
    }
    return NULL;
}

static void clear_profile(struct profile *profile)
{
    size_t i;
    free(profile->id);
    free(profile->name);
    free(profile->description);
    free(profile->ds4_dir);
    free(profile->workdir);
    for (i = 0; i < profile->executable_count; i++)
        free(profile->executables[i]);
    for (i = 0; i < profile->variable_count; i++) {
        free(profile->variables[i].name);
        free(profile->variables[i].label);
        free(profile->variables[i].default_value);
        free(profile->variables[i].value);
        free(profile->variables[i].executable);
    }
    for (i = 0; i < profile->argument_count; i++)
        free(profile->arguments[i]);
    for (i = 0; i < profile->scoped_argument_count; i++) {
        free(profile->scoped_arguments[i].executable);
        free(profile->scoped_arguments[i].value);
    }
    for (i = 0; i < profile->environment_count; i++)
        free(profile->environment[i]);
    memset(profile, 0, sizeof(*profile));
}

static struct profile *start_profile(const char *id)
{
    struct profile *profile = find_profile(id);
    if (profile) {
        clear_profile(profile);
    } else {
        if (profile_count >= MAX_PROFILES)
            die("too many profiles");
        profile = &profiles[profile_count++];
    }
    profile->id = xstrdup(id);
    return profile;
}

static void add_variable(struct profile *profile, const char *spec,
                         const char *executable)
{
    char *copy, *first, *second;
    struct variable *variable;
    if (profile->variable_count >= MAX_VARIABLES)
        die("too many variables in profile");
    copy = xstrdup(spec);
    first = strchr(copy, '|');
    if (!first)
        die("invalid variable; expected NAME|Label|default");
    *first++ = '\0';
    second = strchr(first, '|');
    if (!second)
        die("invalid variable; expected NAME|Label|default");
    *second++ = '\0';
    variable = &profile->variables[profile->variable_count++];
    variable->name = xstrdup(trim(copy));
    variable->label = xstrdup(trim(first));
    variable->default_value = xstrdup(trim(second));
    variable->executable = executable ? xstrdup(executable) : NULL;
    free(copy);
}

static void clear_executables(struct profile *profile)
{
    size_t i;
    for (i = 0; i < profile->executable_count; i++)
        free(profile->executables[i]);
    profile->executable_count = 0;
}

static void add_executable(struct profile *profile, const char *executable)
{
    char *copy;
    if (profile->executable_count >= MAX_EXECUTABLES)
        die("too many executables in profile");
    copy = xstrdup(executable);
    profile->executables[profile->executable_count++] = xstrdup(trim(copy));
    free(copy);
}

static void set_executables(struct profile *profile, const char *spec)
{
    char *copy = xstrdup(spec);
    char *save = NULL;
    char *item;
    clear_executables(profile);
    item = strtok_r(copy, ",", &save);
    while (item) {
        item = trim(item);
        if (*item)
            add_executable(profile, item);
        item = strtok_r(NULL, ",", &save);
    }
    free(copy);
    if (!profile->executable_count)
        die("executables list must not be empty");
}

static void add_scoped_argument(struct profile *profile, const char *executable,
                                const char *value)
{
    struct scoped_argument *argument;
    if (profile->scoped_argument_count >= MAX_SCOPED_ARGUMENTS)
        die("too many executable-specific arguments in profile");
    argument = &profile->scoped_arguments[profile->scoped_argument_count++];
    argument->executable = xstrdup(executable);
    argument->value = xstrdup(value);
}

static void parse_assignment(struct profile *profile, char *line,
                             const char *source, size_t line_number)
{
    char *equals = strchr(line, '=');
    char *key, *value;
    if (!equals) {
        fprintf(stderr, "%s:%zu: expected key=value\n", source, line_number);
        exit(EXIT_FAILURE);
    }
    *equals++ = '\0';
    key = trim(line);
    value = trim(equals);
    if (strcmp(key, "name") == 0) {
        free(profile->name);
        profile->name = xstrdup(value);
    } else if (strcmp(key, "executable") == 0) {
        clear_executables(profile);
        add_executable(profile, value);
    } else if (strcmp(key, "executables") == 0) {
        set_executables(profile, value);
    } else if (strcmp(key, "description") == 0) {
        free(profile->description);
        profile->description = xstrdup(value);
    } else if (strcmp(key, "ds4_dir") == 0) {
        free(profile->ds4_dir);
        profile->ds4_dir = xstrdup(value);
    } else if (strcmp(key, "workdir") == 0) {
        free(profile->workdir);
        profile->workdir = xstrdup(value);
    } else if (strcmp(key, "variable") == 0) {
        add_variable(profile, value, NULL);
    } else if (strncmp(key, "variable.", 9) == 0 && key[9]) {
        add_variable(profile, value, key + 9);
    } else if (strcmp(key, "argument") == 0) {
        if (profile->argument_count >= MAX_ARGUMENTS)
            die("too many arguments in profile");
        profile->arguments[profile->argument_count++] = xstrdup(value);
    } else if (strncmp(key, "argument.", 9) == 0 && key[9]) {
        add_scoped_argument(profile, key + 9, value);
    } else if (strcmp(key, "environment") == 0) {
        if (profile->environment_count >= MAX_ENVIRONMENT)
            die("too many environment entries in profile");
        if (!strchr(value, '='))
            die("environment entry must be NAME=value");
        profile->environment[profile->environment_count++] = xstrdup(value);
    } else {
        fprintf(stderr, "%s:%zu: unknown key '%s'\n", source, line_number, key);
        exit(EXIT_FAILURE);
    }
}

static void parse_profiles_text(const char *text, const char *source)
{
    char *copy = xstrdup(text);
    char *save = NULL;
    char *line = strtok_r(copy, "\n", &save);
    size_t line_number = 0;
    struct profile *current = NULL;
    while (line) {
        char *value;
        line_number++;
        value = trim(line);
        if (*value != '\0' && *value != '#') {
            size_t length = strlen(value);
            if (value[0] == '[' && length > 10 && value[length - 1] == ']') {
                const char prefix[] = "[profile ";
                if (strncmp(value, prefix, sizeof(prefix) - 1) != 0) {
                    fprintf(stderr, "%s:%zu: invalid section\n", source, line_number);
                    exit(EXIT_FAILURE);
                }
                value[length - 1] = '\0';
                current = start_profile(trim(value + sizeof(prefix) - 1));
            } else {
                if (!current) {
                    fprintf(stderr, "%s:%zu: property outside profile\n", source,
                            line_number);
                    exit(EXIT_FAILURE);
                }
                parse_assignment(current, value, source, line_number);
            }
        }
        line = strtok_r(NULL, "\n", &save);
    }
    free(copy);
}

static void load_profiles_file(const char *path, bool required)
{
    FILE *file = fopen(path, "rb");
    char *buffer;
    long size;
    if (!file) {
        if (!required && errno == ENOENT)
            return;
        die_errno(path);
    }
    if (fseek(file, 0, SEEK_END) != 0)
        die_errno(path);
    size = ftell(file);
    if (size < 0 || size > 1024 * 1024)
        die("profile file is invalid or larger than 1 MiB");
    rewind(file);
    buffer = malloc((size_t)size + 1);
    if (!buffer)
        die("out of memory");
    if (fread(buffer, 1, (size_t)size, file) != (size_t)size)
        die_errno(path);
    buffer[size] = '\0';
    fclose(file);
    parse_profiles_text(buffer, path);
    free(buffer);
}

static bool profile_supports_executable(const struct profile *profile,
                                        const char *executable)
{
    size_t i;
    for (i = 0; i < profile->executable_count; i++) {
        if (strcmp(profile->executables[i], executable) == 0)
            return true;
    }
    return false;
}

static void validate_profiles(void)
{
    size_t i;
    for (i = 0; i < profile_count; i++) {
        if (!profiles[i].name || !profiles[i].executable_count)
            die("every profile requires name and executable/executables");
    }
}

static void cleanup_profiles(void)
{
    size_t i;
    for (i = 0; i < profile_count; i++)
        clear_profile(&profiles[i]);
    profile_count = 0;
}

static void cleanup_options(struct options *options)
{
    size_t i;
    for (i = 0; i < options->override_count; i++) {
        free(options->overrides[i].name);
        free(options->overrides[i].value);
    }
    options->override_count = 0;
}

static void print_usage(FILE *stream)
{
    fprintf(stream,
        "Usage: ds4-launcher [EXECUTABLE] [options] [-- additional DS4 arguments]\n\n"
        "Interactive mode (no --profile) lets you choose an available DS4\n"
        "executable, then a ready profile, then edit per-run values.\n\n"
        "Options:\n"
        "  EXECUTABLE             ds4, ds4-server, ds4-bench, ds4-eval, or ds4-agent\n"
        "  -p, --profile ID       Select a hardware/model configuration\n"
        "  -e, --executable NAME  Legacy/automation alternative to EXECUTABLE\n"
        "  -D, --ds4-dir DIR      Directory containing DS4 executables/assets\n"
        "  -C, --workdir DIR      Working directory used while running DS4\n"
        "  -s, --set NAME=VALUE   Override a profile variable (repeatable)\n"
        "      --profiles FILE    Load/override profiles from a file\n"
        "  -l, --list             List profiles\n"
        "      --list-executables List detected DS4 executables\n"
        "  -n, --dry-run          Print the command without executing it\n"
        "      --no-prompt        Use defaults and --set values without prompts\n"
        "  -y, --yes              Skip final confirmation\n"
        "  -h, --help             Show this help\n\n"
        "Examples:\n"
        "  ds4-launcher\n"
        "  ds4-launcher --list\n"
        "  ds4-launcher ds4-server -p glm52-dgx-strix\n"
        "  ds4-launcher ds4 -p glm-worker-strix --set COORDINATOR_IP=10.44.0.9\n");
}

static void add_override(struct options *options, const char *spec)
{
    char *copy, *equals;
    if (options->override_count >= MAX_OVERRIDES)
        die("too many --set overrides");
    copy = xstrdup(spec);
    equals = strchr(copy, '=');
    if (!equals || equals == copy)
        die("--set expects NAME=VALUE");
    *equals++ = '\0';
    options->overrides[options->override_count].name = xstrdup(copy);
    options->overrides[options->override_count].value = xstrdup(equals);
    options->override_count++;
    free(copy);
}

static bool is_known_executable(const char *value)
{
    size_t i;
    for (i = 0; i < sizeof(known_executables) / sizeof(known_executables[0]); i++) {
        if (strcmp(value, known_executables[i]) == 0)
            return true;
    }
    return false;
}

static void set_requested_executable(struct options *options, const char *value)
{
    if (!is_known_executable(value)) {
        fprintf(stderr, "ds4-launcher: unknown executable '%s'\n", value);
        exit(EXIT_FAILURE);
    }
    if (options->executable && strcmp(options->executable, value) != 0)
        die("conflicting executable selections");
    options->executable = value;
}

static void parse_options(int argc, char **argv, struct options *options)
{
    int i;
    memset(options, 0, sizeof(*options));
    for (i = 1; i < argc; i++) {
        const char *arg = argv[i];
        if (strcmp(arg, "--") == 0) {
            i++;
            for (; i < argc; i++) {
                if (options->extra_arg_count >= MAX_EXTRA_ARGS)
                    die("too many additional arguments");
                options->extra_args[options->extra_arg_count++] = argv[i];
            }
            break;
        } else if (strcmp(arg, "-h") == 0 || strcmp(arg, "--help") == 0) {
            options->show_help = true;
        } else if (strcmp(arg, "-l") == 0 || strcmp(arg, "--list") == 0) {
            options->list_profiles = true;
        } else if (strcmp(arg, "--list-executables") == 0) {
            options->list_executables = true;
        } else if (strcmp(arg, "-n") == 0 || strcmp(arg, "--dry-run") == 0) {
            options->dry_run = true;
        } else if (strcmp(arg, "--no-prompt") == 0) {
            options->no_prompt = true;
        } else if (strcmp(arg, "-y") == 0 || strcmp(arg, "--yes") == 0) {
            options->assume_yes = true;
        } else if (strcmp(arg, "-p") == 0 || strcmp(arg, "--profile") == 0) {
            if (++i >= argc)
                die("--profile requires an ID");
            options->profile_id = argv[i];
        } else if (strcmp(arg, "-e") == 0 || strcmp(arg, "--executable") == 0) {
            if (++i >= argc)
                die("--executable requires a name");
            set_requested_executable(options, argv[i]);
        } else if (strcmp(arg, "-D") == 0 || strcmp(arg, "--ds4-dir") == 0) {
            if (++i >= argc)
                die("--ds4-dir requires a directory");
            options->ds4_dir = argv[i];
        } else if (strcmp(arg, "-C") == 0 || strcmp(arg, "--workdir") == 0) {
            if (++i >= argc)
                die("--workdir requires a directory");
            options->workdir = argv[i];
        } else if (strcmp(arg, "--profiles") == 0) {
            if (++i >= argc)
                die("--profiles requires a file");
            options->profiles_file = argv[i];
        } else if (strcmp(arg, "-s") == 0 || strcmp(arg, "--set") == 0) {
            if (++i >= argc)
                die("--set requires NAME=VALUE");
            add_override(options, argv[i]);
        } else if (arg[0] != '-' && is_known_executable(arg)) {
            set_requested_executable(options, arg);
        } else {
            fprintf(stderr, "ds4-launcher: unknown option or executable: %s\n", arg);
            print_usage(stderr);
            exit(EXIT_FAILURE);
        }
    }
}

static bool executable_file(const char *path)
{
    struct stat info;
    return access(path, X_OK) == 0 && stat(path, &info) == 0 && S_ISREG(info.st_mode);
}

static bool join_path(char *output, size_t size, const char *left, const char *right)
{
    int written = snprintf(output, size, "%s%s%s", left,
                           left[0] && left[strlen(left) - 1] == '/' ? "" : "/", right);
    return written >= 0 && (size_t)written < size;
}

static bool absolute_path(const char *path, char *output, size_t size)
{
    char resolved[PATH_MAX];
    if (!realpath(path, resolved))
        return false;
    if (strlen(resolved) + 1 > size)
        return false;
    strcpy(output, resolved);
    return true;
}

static bool find_on_path(const char *name, char *output, size_t size)
{
    const char *path_value = getenv("PATH");
    char *copy, *save = NULL, *entry;
    if (!path_value)
        return false;
    copy = xstrdup(path_value);
    entry = strtok_r(copy, ":", &save);
    while (entry) {
        char candidate[PATH_MAX];
        if (join_path(candidate, sizeof(candidate), *entry ? entry : ".", name) &&
            executable_file(candidate) && absolute_path(candidate, output, size)) {
            free(copy);
            return true;
        }
        entry = strtok_r(NULL, ":", &save);
    }
    free(copy);
    return false;
}

static bool resolve_executable(const char *name, const char *ds4_dir,
                               char *output, size_t size)
{
    char candidate[PATH_MAX];
    const char *home;
    if (ds4_dir) {
        if (!join_path(candidate, sizeof(candidate), ds4_dir, name))
            return false;
        return executable_file(candidate) && absolute_path(candidate, output, size);
    }
    if (join_path(candidate, sizeof(candidate), ".", name) && executable_file(candidate) &&
        absolute_path(candidate, output, size))
        return true;
    home = getenv("HOME");
    if (home && snprintf(candidate, sizeof(candidate), "%s/ds4/%s", home, name) > 0 &&
        executable_file(candidate) && absolute_path(candidate, output, size))
        return true;
    return find_on_path(name, output, size);
}

static void list_profiles(void)
{
    size_t i;
    const char *last = NULL;
    for (i = 0; i < sizeof(known_executables) / sizeof(known_executables[0]); i++) {
        size_t p;
        last = known_executables[i];
        printf("%s:\n", last);
        for (p = 0; p < profile_count; p++) {
            if (profile_supports_executable(&profiles[p], last))
                printf("  %-30s %s\n", profiles[p].id, profiles[p].name);
        }
    }
}

static void list_executables(const char *ds4_dir)
{
    size_t i;
    for (i = 0; i < sizeof(known_executables) / sizeof(known_executables[0]); i++) {
        char path[PATH_MAX];
        bool found = resolve_executable(known_executables[i], ds4_dir, path, sizeof(path));
        printf("%-12s %s%s\n", known_executables[i], found ? path : "not found",
               found ? "" : "");
    }
}

static int read_selection(size_t maximum)
{
    char line[64], *end;
    long selected;
    while (true) {
        printf("Choice [1-%zu]: ", maximum);
        fflush(stdout);
        if (!fgets(line, sizeof(line), stdin))
            die("input ended");
        errno = 0;
        selected = strtol(line, &end, 10);
        while (isspace((unsigned char)*end))
            end++;
        if (!errno && *end == '\0' && selected >= 1 && (size_t)selected <= maximum)
            return (int)selected - 1;
        fprintf(stderr, "Please enter a number from 1 to %zu.\n", maximum);
    }
}

static const char *select_executable(const char *ds4_dir)
{
    const char *available[sizeof(known_executables) / sizeof(known_executables[0])];
    size_t count = 0, i;
    printf("Available DS4 executables:\n");
    for (i = 0; i < sizeof(known_executables) / sizeof(known_executables[0]); i++) {
        char path[PATH_MAX];
        if (resolve_executable(known_executables[i], ds4_dir, path, sizeof(path))) {
            available[count++] = known_executables[i];
            printf("  %zu) %-12s %s\n", count, known_executables[i], path);
        }
    }
    if (!count)
        die("no DS4 executables found; use --ds4-dir DIR");
    return available[read_selection(count)];
}

static const char *select_profile_executable(const struct profile *profile,
                                             const char *ds4_dir)
{
    const char *available[MAX_EXECUTABLES];
    size_t count = 0, i;
    printf("\nExecutables supported by %s:\n", profile->name);
    for (i = 0; i < profile->executable_count; i++) {
        char path[PATH_MAX];
        char *profile_ds4_dir = NULL;
        const char *effective_ds4_dir = ds4_dir;
        if (!effective_ds4_dir && profile->ds4_dir) {
            profile_ds4_dir = expand_value(profile->ds4_dir, profile,
                                            profile->executables[i]);
            effective_ds4_dir = profile_ds4_dir;
        }
        if (resolve_executable(profile->executables[i], effective_ds4_dir,
                               path, sizeof(path))) {
            available[count++] = profile->executables[i];
            printf("  %zu) %-12s %s\n", count, profile->executables[i], path);
        }
        free(profile_ds4_dir);
    }
    if (!count)
        die("none of the profile executables were found");
    if (count == 1)
        return available[0];
    return available[read_selection(count)];
}

static struct profile *select_profile(const char *executable)
{
    struct profile *available[MAX_PROFILES];
    size_t count = 0, i;
    printf("\nProfiles for %s:\n", executable);
    for (i = 0; i < profile_count; i++) {
        if (profile_supports_executable(&profiles[i], executable)) {
            available[count++] = &profiles[i];
            printf("  %zu) %s\n     %s\n", count, profiles[i].name,
                   profiles[i].description ? profiles[i].description : "");
        }
    }
    if (!count)
        die("no profiles available for the selected executable");
    return available[read_selection(count)];
}

static bool variable_applies(const struct variable *variable,
                             const char *executable)
{
    return !variable->executable || strcmp(variable->executable, executable) == 0;
}

static const char *profile_value(const struct profile *profile, const char *name,
                                 const char *executable)
{
    size_t i;
    for (i = 0; i < profile->variable_count; i++) {
        if (profile->variables[i].executable &&
            strcmp(profile->variables[i].executable, executable) == 0 &&
            strcmp(profile->variables[i].name, name) == 0)
            return profile->variables[i].value ? profile->variables[i].value :
                                                 profile->variables[i].default_value;
    }
    for (i = 0; i < profile->variable_count; i++) {
        if (!profile->variables[i].executable &&
            strcmp(profile->variables[i].name, name) == 0)
            return profile->variables[i].value ? profile->variables[i].value :
                                                 profile->variables[i].default_value;
    }
    return getenv(name);
}

static char *expand_once(const char *input, const struct profile *profile,
                         const char *executable, bool *changed)
{
    size_t capacity = strlen(input) + 64;
    size_t used = 0, i = 0;
    char *output = malloc(capacity);
    if (!output)
        die("out of memory");
    *changed = false;
    while (input[i]) {
        const char *replacement = NULL;
        size_t consumed = 1;
        char name[128];
        if (input[i] == '$' && input[i + 1] == '{') {
            const char *close = strchr(input + i + 2, '}');
            if (close && (size_t)(close - (input + i + 2)) < sizeof(name)) {
                size_t length = (size_t)(close - (input + i + 2));
                memcpy(name, input + i + 2, length);
                name[length] = '\0';
                replacement = profile_value(profile, name, executable);
                consumed = (size_t)(close - (input + i)) + 1;
                if (!replacement) {
                    fprintf(stderr, "ds4-launcher: undefined variable ${%s}\n", name);
                    exit(EXIT_FAILURE);
                }
                *changed = true;
            }
        }
        if (replacement) {
            size_t length = strlen(replacement);
            while (used + length + 1 > capacity) {
                capacity *= 2;
                output = realloc(output, capacity);
                if (!output)
                    die("out of memory");
            }
            memcpy(output + used, replacement, length);
            used += length;
            i += consumed;
        } else {
            if (used + 2 > capacity) {
                capacity *= 2;
                output = realloc(output, capacity);
                if (!output)
                    die("out of memory");
            }
            output[used++] = input[i++];
        }
    }
    output[used] = '\0';
    return output;
}

static char *expand_value(const char *input, const struct profile *profile,
                          const char *executable)
{
    char *current = xstrdup(input);
    int pass;
    for (pass = 0; pass < 8; pass++) {
        bool changed;
        char *next = expand_once(current, profile, executable, &changed);
        free(current);
        current = next;
        if (!changed)
            break;
    }
    if (current[0] == '~' && current[1] == '/') {
        const char *home = getenv("HOME");
        if (home) {
            size_t size = strlen(home) + strlen(current);
            char *expanded = malloc(size);
            if (!expanded)
                die("out of memory");
            snprintf(expanded, size, "%s%s", home, current + 1);
            free(current);
            current = expanded;
        }
    }
    return current;
}

static struct variable *find_variable(struct profile *profile, const char *name,
                                      const char *executable)
{
    size_t i;
    for (i = 0; i < profile->variable_count; i++) {
        if (profile->variables[i].executable &&
            strcmp(profile->variables[i].executable, executable) == 0 &&
            strcmp(profile->variables[i].name, name) == 0)
            return &profile->variables[i];
    }
    for (i = 0; i < profile->variable_count; i++) {
        if (!profile->variables[i].executable &&
            strcmp(profile->variables[i].name, name) == 0)
            return &profile->variables[i];
    }
    return NULL;
}

static void apply_overrides(struct profile *profile, const struct options *options,
                            const char *executable)
{
    size_t i;
    for (i = 0; i < options->override_count; i++) {
        struct variable *variable = find_variable(profile, options->overrides[i].name,
                                                  executable);
        if (!variable) {
            fprintf(stderr, "ds4-launcher: profile '%s' has no variable '%s'\n",
                    profile->id, options->overrides[i].name);
            exit(EXIT_FAILURE);
        }
        free(variable->value);
        variable->value = xstrdup(options->overrides[i].value);
        variable->supplied = true;
    }
}

static void prompt_variables(struct profile *profile, const char *executable)
{
    size_t i;
    char line[MAX_LINE];
    printf("\nProfile: %s\n%s\n\n", profile->name,
           profile->description ? profile->description : "");
    for (i = 0; i < profile->variable_count; i++) {
        struct variable *variable = &profile->variables[i];
        char *default_value;
        if (!variable_applies(variable, executable) || variable->supplied)
            continue;
        default_value = expand_value(variable->value ? variable->value :
                                                       variable->default_value,
                                     profile, executable);
        printf("%s [%s]: ", variable->label, default_value);
        fflush(stdout);
        if (!fgets(line, sizeof(line), stdin))
            die("input ended");
        line[strcspn(line, "\r\n")] = '\0';
        free(variable->value);
        variable->value = xstrdup(*line ? line : default_value);
        free(default_value);
    }
}

static void shell_quote(FILE *stream, const char *value)
{
    const char *p;
    bool safe = *value != '\0';
    for (p = value; *p; p++) {
        if (!(isalnum((unsigned char)*p) || strchr("_@%+=:,./-", *p))) {
            safe = false;
            break;
        }
    }
    if (safe) {
        fputs(value, stream);
        return;
    }
    fputc('\'', stream);
    for (p = value; *p; p++) {
        if (*p == '\'')
            fputs("'\\''", stream);
        else
            fputc(*p, stream);
    }
    fputc('\'', stream);
}

static bool confirm_run(void)
{
    char line[32];
    printf("\nExecute this command? [Y/n]: ");
    fflush(stdout);
    if (!fgets(line, sizeof(line), stdin))
        return false;
    return line[0] == '\0' || line[0] == '\n' || line[0] == 'y' || line[0] == 'Y';
}

static const char *derive_workdir(const struct options *options,
                                  const char *profile_workdir,
                                  const char *effective_ds4_dir,
                                  const char *executable_path,
                                  char *buffer, size_t size)
{
    const char *slash;
    if (options->workdir)
        return options->workdir;
    if (options->ds4_dir)
        return options->ds4_dir;
    if (profile_workdir)
        return profile_workdir;
    if (effective_ds4_dir)
        return effective_ds4_dir;
    slash = strrchr(executable_path, '/');
    if (slash) {
        size_t length = (size_t)(slash - executable_path);
        if (length < size && length > 0) {
            memcpy(buffer, executable_path, length);
            buffer[length] = '\0';
            if (strcmp(buffer, "/usr/bin") != 0 && strcmp(buffer, "/usr/local/bin") != 0)
                return buffer;
        }
    }
    if (!getcwd(buffer, size))
        die_errno("getcwd");
    return buffer;
}

static void free_command(char **argv, size_t argc, char **environment,
                         size_t environment_count)
{
    size_t i;
    for (i = 1; i < argc; i++)
        free(argv[i]);
    for (i = 0; i < environment_count; i++)
        free(environment[i]);
}

static int run_profile(struct profile *profile, const char *executable,
                       const struct options *options)
{
    char executable_path[PATH_MAX], workdir_buffer[PATH_MAX];
    char *profile_ds4_dir = NULL;
    char *profile_workdir = NULL;
    const char *effective_ds4_dir = options->ds4_dir;
    const char *workdir;
    char *argv[MAX_ARGUMENTS + MAX_SCOPED_ARGUMENTS + MAX_EXTRA_ARGS + 2];
    char *expanded_env[MAX_ENVIRONMENT];
    size_t argc = 0, i;

    if (!effective_ds4_dir && profile->ds4_dir) {
        profile_ds4_dir = expand_value(profile->ds4_dir, profile, executable);
        effective_ds4_dir = profile_ds4_dir;
    }
    if (!resolve_executable(executable, effective_ds4_dir,
                            executable_path, sizeof(executable_path))) {
        fprintf(stderr, "ds4-launcher: executable '%s' not found; use --ds4-dir DIR\n",
                executable);
        free(profile_ds4_dir);
        return EXIT_FAILURE;
    }
    if (!options->workdir && !options->ds4_dir && profile->workdir)
        profile_workdir = expand_value(profile->workdir, profile, executable);
    workdir = derive_workdir(options, profile_workdir, effective_ds4_dir,
                             executable_path, workdir_buffer,
                             sizeof(workdir_buffer));
    argv[argc++] = executable_path;
    for (i = 0; i < profile->argument_count; i++)
        argv[argc++] = expand_value(profile->arguments[i], profile, executable);
    for (i = 0; i < profile->scoped_argument_count; i++) {
        if (strcmp(profile->scoped_arguments[i].executable, executable) == 0)
            argv[argc++] = expand_value(profile->scoped_arguments[i].value, profile,
                                        executable);
    }
    for (i = 0; i < options->extra_arg_count; i++)
        argv[argc++] = xstrdup(options->extra_args[i]);
    argv[argc] = NULL;

    for (i = 0; i < profile->environment_count; i++)
        expanded_env[i] = expand_value(profile->environment[i], profile, executable);

    printf("\nWorking directory: ");
    shell_quote(stdout, workdir);
    printf("\nCommand:\n  ");
    for (i = 0; i < profile->environment_count; i++) {
        shell_quote(stdout, expanded_env[i]);
        fputc(' ', stdout);
    }
    for (i = 0; i < argc; i++) {
        if (i)
            fputc(' ', stdout);
        shell_quote(stdout, argv[i]);
    }
    fputc('\n', stdout);

    if (options->dry_run) {
        free_command(argv, argc, expanded_env, profile->environment_count);
        free(profile_workdir);
        free(profile_ds4_dir);
        return EXIT_SUCCESS;
    }
    if (!options->assume_yes && !options->no_prompt && !confirm_run()) {
        puts("Cancelled.");
        free_command(argv, argc, expanded_env, profile->environment_count);
        free(profile_workdir);
        free(profile_ds4_dir);
        return EXIT_SUCCESS;
    }
    for (i = 0; i < profile->environment_count; i++) {
        char *equals = strchr(expanded_env[i], '=');
        if (!equals)
            die("internal invalid environment assignment");
        *equals++ = '\0';
        if (setenv(expanded_env[i], equals, 1) != 0)
            die_errno("setenv");
    }
    if (chdir(workdir) != 0)
        die_errno(workdir);
    execv(executable_path, argv);
    die_errno(executable_path);
    return EXIT_FAILURE;
}

static void load_default_profile_files(const struct options *options)
{
    char path[PATH_MAX];
    const char *home;
    parse_profiles_text(builtin_profiles, "built-in profiles");
    load_profiles_file("/etc/ds4-launcher/profiles.conf", false);
    home = getenv("HOME");
    if (home && snprintf(path, sizeof(path), "%s/.config/ds4-launcher/profiles.conf", home) > 0)
        load_profiles_file(path, false);
    if (options->profiles_file)
        load_profiles_file(options->profiles_file, true);
    validate_profiles();
}

int main(int argc, char **argv)
{
    struct options options;
    struct profile *profile;
    const char *executable;
    const char *environment_ds4_dir;
    int result;

    parse_options(argc, argv, &options);
    if (options.show_help) {
        print_usage(stdout);
        cleanup_options(&options);
        return EXIT_SUCCESS;
    }
    environment_ds4_dir = getenv("DS4_DIR");
    if (!options.ds4_dir && environment_ds4_dir && *environment_ds4_dir)
        options.ds4_dir = environment_ds4_dir;
    load_default_profile_files(&options);
    if (options.list_profiles) {
        list_profiles();
        cleanup_profiles();
        cleanup_options(&options);
        return EXIT_SUCCESS;
    }
    if (options.list_executables) {
        list_executables(options.ds4_dir);
        cleanup_profiles();
        cleanup_options(&options);
        return EXIT_SUCCESS;
    }
    if (options.profile_id) {
        profile = find_profile(options.profile_id);
        if (!profile) {
            fprintf(stderr, "ds4-launcher: unknown profile '%s'\n", options.profile_id);
            cleanup_profiles();
            cleanup_options(&options);
            return EXIT_FAILURE;
        }
        if (options.executable) {
            if (!profile_supports_executable(profile, options.executable)) {
                fprintf(stderr, "ds4-launcher: profile '%s' does not support '%s'\n",
                        profile->id, options.executable);
                cleanup_profiles();
                cleanup_options(&options);
                return EXIT_FAILURE;
            }
            executable = options.executable;
        } else if (profile->executable_count == 1) {
            executable = profile->executables[0];
        } else {
            if (!isatty(STDIN_FILENO))
                die("multi-executable profile requires --executable NAME");
            executable = select_profile_executable(profile, options.ds4_dir);
        }
    } else {
        if (!isatty(STDIN_FILENO))
            die("interactive selection requires a terminal; use --profile ID");
        executable = options.executable ? options.executable :
                                          select_executable(options.ds4_dir);
        profile = select_profile(executable);
    }
    apply_overrides(profile, &options, executable);
    if (!options.no_prompt) {
        if (!isatty(STDIN_FILENO))
            die("variable prompts require a terminal; use --no-prompt");
        prompt_variables(profile, executable);
    }
    result = run_profile(profile, executable, &options);
    cleanup_profiles();
    cleanup_options(&options);
    return result;
}
