# Usage: 
#   Replace the direct_register_custom_op function in torch_utils.py with the patched version below,
#   then run the script and execute the target code within a CaptureMode context, e.g.:
#   with CaptureMode():
#       y = model(x)
#   And the operators and all input/output tensors will be saved to the logs/ directory for inspection.

import torch
from torch.utils._python_dispatch import TorchDispatchMode
import os
import typing
import contextlib
import torch.nn as nn

class CaptureMode(TorchDispatchMode):
    _instance = None

    def _is_cpp(name: str) -> bool:
        table = torch._C._dispatch_dump_table(name)
        backends = [line.split(": ")[0] for line in table.splitlines()]
        return "CUDA" in backends

    def __init__(self, log_dir="./logs"):
        super().__init__()
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.tensor_serial = 0
        self.operator_serial = 0
        self.suppressed = False

    def _save_tensor(self, name: str, tensor: torch.Tensor):
        tensor_id = self.tensor_serial
        self.tensor_serial += 1
        type = 'x'.join(list(str(x) for x in tensor.shape) + [str(tensor.dtype).split('.')[-1]])
        filename = f"{tensor_id:05d}-{type}-{name}.bin"
        path = os.path.join(self.log_dir, filename)
        with open(path, 'wb') as f:
            f.write(bytes(tensor.detach().cpu().untyped_storage()))
    
    def _enter_dir(self, name: str):
        self.log_dir = os.path.join(self.log_dir, name)
        os.makedirs(self.log_dir, exist_ok=True)

    def _leave_dir(self):
        self.log_dir = os.path.dirname(self.log_dir)

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if self.suppressed:
            return func(*args, **kwargs)
        schema: torch.FunctionSchema = func._schema
        operator_name = schema.name
        if not CaptureMode._is_cpp(operator_name):
            return func(*args, **kwargs)
        operator_overload_name = schema.overload_name
        operator_id = self.operator_serial
        self.operator_serial += 1
        self._enter_dir(f"{operator_id:04d}-{operator_name}{'.' if operator_overload_name else ''}{operator_overload_name}")
        args_ptr = 0
        args_dict = {}
        outputs_dict = {}
        for arg in schema.arguments:
            if kwargs is not None and arg.name in kwargs:
                arg_value = kwargs[arg.name]
            elif not arg.kwarg_only and args_ptr < len(args):
                arg_value = args[args_ptr]
                args_ptr += 1
            else:
                continue
            if isinstance(arg_value, torch.Tensor):
                self._save_tensor(arg.name, arg_value)
                if arg.is_write:
                    outputs_dict[arg.name] = arg_value
            elif isinstance(arg_value, (list, tuple)) and len(arg_value) > 0 and isinstance(arg_value[0], torch.Tensor):
                for (i, item) in enumerate(arg_value):
                    if isinstance(item, torch.Tensor):
                        self._save_tensor(f"{arg.name}.{i}", item)
                        if arg.is_write:
                            outputs_dict[f"{arg.name}.{i}"] = item
            else:
                args_dict[arg.name] = repr(arg_value)
        
        self.suppressed = True
        out = func(*args, **kwargs)
        self.suppressed = False

        for name, tensor in outputs_dict.items():
            self._save_tensor(f"{name}-output", tensor)
    
        if isinstance(out, torch.Tensor):
            self._save_tensor("return", out)
        elif isinstance(out, (list, tuple)) and len(out) > 0 and isinstance(out[0], torch.Tensor):
            for (i, item) in enumerate(out):
                if isinstance(item, torch.Tensor):
                    self._save_tensor(f"return.{i}", item)
        else:
            args_dict["return"] = repr(out)

        with open(os.path.join(self.log_dir, "schema.txt"), "w") as f:
            f.write(repr(schema))

        with open(os.path.join(self.log_dir, "scalars.txt"), "w") as f:
            for name, value in args_dict.items():
                f.write(f"{name}: {value}\n")
        self._leave_dir()
        return out

    def __enter__(self):
        self._previous_instance = self.__class__._instance
        self.__class__._instance = self
        return super().__enter__()
    
    def __exit__(self, exc_type, exc_value, traceback):
        assert self.__class__._instance is self
        self.__class__._instance = self._previous_instance
        return super().__exit__(exc_type, exc_value, traceback)

    @contextlib.contextmanager
    def group(name: str):
        instance = CaptureMode._instance
        if instance is not None:
            instance._enter_dir(name)
        yield
        if instance is not None:
            instance._leave_dir()

    def autogroup(module: nn.Module):
        def _make_hook(module: nn.Module, name: str):
            if hasattr(module, "_capture_hooked"):
                return
            module._capture_hooked = True
            module.register_forward_pre_hook(lambda module, input: CaptureMode._instance._enter_dir(name))
            module.register_forward_hook(lambda module, input, output: CaptureMode._instance._leave_dir())
            for child_name, child in module.named_children():
                _make_hook(child, child_name)
        return _make_hook(module, "")

def direct_register_custom_op_patched(
    op_name: str,
    op_func: Callable,
    mutates_args: list[str] | None = None,
    fake_impl: Callable | None = None,
    target_lib: Library | None = None,
    dispatch_key: str | None = None,
    tags: tuple[torch.Tag, ...] = (),
):
    from torch.utils._python_dispatch import TorchDispatchMode
    from vllm.log_ops import CaptureMode

    def wrapper(*args, **kwargs):
        if CaptureMode._instance is not None:
            with CaptureMode._instance:
                return op_func(*args, **kwargs)
        else:
            return op_func(*args, **kwargs)

    if mutates_args is None:
        mutates_args = []

    if dispatch_key is None:
        from vllm.platforms import current_platform

        dispatch_key = current_platform.dispatch_key

    schema_str = infer_schema(op_func, mutates_args=mutates_args)

    my_lib = target_lib or vllm_lib
    my_lib.define(op_name + schema_str, tags=tags)
    my_lib.impl(op_name, wrapper, dispatch_key=dispatch_key)
    if fake_impl is not None:
        my_lib._register_fake(op_name, fake_impl)
