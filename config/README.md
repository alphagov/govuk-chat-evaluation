# GOV.UK Chat Evaluation

## Config

This directory is for storing config files used for evaluations.

They can be specified by providing a `config_path` argument to a task, for example: `uv run govuk_chat_evaluation question_router --config_path config/my_custom_config.yaml`. Each task has a default config that is applied automatically if you don't specify one.

The ones in the default directory are committed to the repository whereas other files within it, or within other directories, will only be available on your local system.

Make changes to the default files if you want to affect how someone typically runs an evaluation or for a common evaluation scenario. For your own specific testing, use this config directory.

A couple of usage tips:

- simple options in config files can typically be overridden with command line arguments when running a task, often reducing the need to create new config files (for example, a task with `generate: true` can be overridden with a `--no-generate` CLI option) - run `--help` against the task for options
- whenever an evaluation is run a config file for that generation is stored in the results directory and can be used again to repeat the same configuration.

### Configuring the evaluation model

You can configure the model when you are running an evaluation related to answer composition.

The model must be one that is supported by the Ruby codebase. You can determine which models are supported by checking the [SUPPORTED_MODELS constant](https://github.com/search?q=repo%3Aalphagov%2Fgovuk-chat+SUPPORTED_MODELS&type=code) for the class related to the type of evaluation you are trying to run.

Each of these supported models has its own prompt configuration in the [private repo](https://github.com/alphagov/govuk_chat_private/tree/main/config/llm_prompts/claude).

Once you have established which Claude model you want to use to run your evaluation, you can set the `claude_generation_model` in the config file or pass it via a command line argument. For example:
`uv run govuk_chat_evaluation question_router --claude_generation_model claude_haiku_4_5 --generate`

The model will be passed to the Ruby codebase as an environment variable, and that model will be used in subsequent LLM calls that are used to generate the evaluation inputs.

#### Running an evaluation using an unsupported Claude model

There may be situations where you want to run an evaluation using an unsupported model, for example when testing a new version of a model. To do this, you will need to make the following changes to your local environment:

1. Add the prompt configuration for that model to the private repo. The model key should use the following format: `claude_<model-name>_<version>`.Use underscores instead of dots in the version number. For example: `claude_sonnet_5_0`.
2. Update the private repo gem to point to your local repo by changing
   `gem "govuk_chat_private", github: "alphagov/govuk_chat_private"`
   to
   `gem "govuk_chat_private", path: "../govuk_chat_private"`
   (or the appropriate path to your private repo).
3. Add the model to the `MODEL_IDS` constant in the [BedrockModels module](https://github.com/alphagov/govuk-chat/blob/806a05dc9093d7c1ba2089086576e6a1094f484e/lib/bedrock_models.rb#L1). The key used should match the model key used in step 1.
4. Update the array in the [SUPPORTED_MODELS constant](https://github.com/search?q=repo%3Aalphagov%2Fgovuk-chat+SUPPORTED_MODELS&type=code) for the relevant component to include the new model.
5. Follow the guidance above on updating the configuration or passing the model via a CLI argument.
