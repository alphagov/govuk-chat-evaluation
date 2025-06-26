from inspect import isclass
from pathlib import Path
from typing import (
    Annotated,
    Any,
    Literal,
    Optional,
    Self,
    Type,
    TypeVar,
    get_args,
    get_origin,
)

import click
import yaml
from pydantic import BaseModel, Field, FilePath

GenericConfig = TypeVar("GenericConfig", bound="BaseConfig")


class BaseConfig(BaseModel):
    class GenericFields:
        """Commonly used fields across Configs"""

        what = Annotated[str, Field(..., description="What is being evaluated")]
        generate = Annotated[bool, Field(..., description="Whether to generate data")]
        provider_openai_or_claude = Annotated[
            Optional[Literal["openai", "claude"]],
            Field(
                None,
                description="Which provider to use for generating the data, openai or claude",
            ),
        ]
        input_path = Annotated[
            FilePath, Field(..., description="Path to the data file used to evaluate")
        ]

        provider_openai_or_titan: Annotated[
            str,
            Field(..., pattern="^(openai|titan)$", description="Which embedding engine to use"),
        ]

    def _validate_fields_required_for_generate(self, *fields) -> Self:
        if getattr(self, "generate", False):
            for field in fields:
                if hasattr(self, field) and getattr(self, field, None) is None:
                    raise ValueError(f"{field} is required to generate data")

        return self

    @classmethod
    def apply_click_options(cls, command):
        for field_name, field_info in cls.model_fields.items():
            description = field_info.description

            field_type = field_info.annotation

            if get_origin(field_type) is Optional:
                field_type = get_args(field_type)[0]

            if field_type is bool:
                command = click.option(
                    f"--{field_name}/--no-{field_name}", help=description, default=None
                )(command)
            elif (
                # Try avoid complex types such as lists and nested objects
                get_origin(field_type) not in {list, dict}
                and not (isclass(field_type) and issubclass(field_type, BaseModel))
            ):
                command = click.option(f"--{field_name}", help=description)(command)

        return command


def apply_click_options_to_command(config_cls: Type[GenericConfig]):
    def decorator(command):
        return config_cls.apply_click_options(command)

    return decorator


def config_from_cli_args(
    config_path: Path, config_cls: Type[GenericConfig], cli_args: dict[str, Any]
) -> GenericConfig:
    filtered_args = {k: v for k, v in cli_args.items() if v is not None}

    with open(config_path, "r") as file:
        config_data = yaml.safe_load(file)

    return config_cls(**(config_data | filtered_args))
