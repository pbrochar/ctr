import ast
from functools import wraps
from typing import Callable, Optional, List
import inspect


class BaseContractValidationError(Exception):
    def __init__(self, message):
        self.message = message


class ContractTypeError(BaseContractValidationError):
    pass


class MissingAnnotationError(ContractTypeError):
    pass


class ContractPreConditionError(BaseContractValidationError):
    pass


class ContractPostConditionError(BaseContractValidationError):
    pass


class CTRDocstringParser:
    BASE_DOCSTRING_CODE_BLOCK = "ctr"
    PRECONDITION_DOCSTRING_CODE_BLOCK = "pre"
    POSTCONDITION_DOCSTRING_CODE_BLOCK = "post"

    def __init__(self, docstring: str):
        self.docstring = docstring
        self._splitted_docstring = self.docstring.split("\n")
        self.pre_assertions = self._extract_pre_ctr_block(self._splitted_docstring)
        self.post_assertions = self._extract_post_ctr_block(self._splitted_docstring)

    def _extract_pre_ctr_block(self, docstring: list[str]) -> list[str]:
        precondition_block_name = f"{self.BASE_DOCSTRING_CODE_BLOCK}.{self.PRECONDITION_DOCSTRING_CODE_BLOCK}:"
        pre_condition_block = self._extract_ctr_block(
            docstring, precondition_block_name
        )
        stripped_assertions = self._get_striped_assertions(pre_condition_block)
        return self._get_valid_python_assertions(stripped_assertions)

    def _extract_post_ctr_block(self, docstring: list[str]) -> list[str]:
        postcondition_block_name = f"{self.BASE_DOCSTRING_CODE_BLOCK}.{self.POSTCONDITION_DOCSTRING_CODE_BLOCK}:"
        post_condition_block = self._extract_ctr_block(
            docstring, postcondition_block_name
        )
        stripped_assertions = self._get_striped_assertions(post_condition_block)
        return self._get_valid_python_assertions(stripped_assertions)

    @staticmethod
    def _extract_ctr_block(docstring: list[str], code_block: str):
        ctr_block = []
        inside_ctr = False

        for line in docstring:
            if inside_ctr:
                if line.startswith(" "):  # Check if the line is indented
                    ctr_block.append(line.strip())
                else:
                    break  # Exit the block if the indentation ends
            elif line.strip() == code_block:
                inside_ctr = True

        return ctr_block

    @staticmethod
    def _get_striped_assertions(docs: list[str]) -> list[str]:
        valid_assertions = []
        for i, doc in enumerate(docs):
            if not isinstance(doc, str):
                continue
            if not doc:
                docs.pop(i)
            valid_assertions.append(doc.strip())

        return valid_assertions

    @staticmethod
    def _is_valid_python_code(code: str):
        try:
            ast.parse(code)
            return True
        except SyntaxError:
            return False

    def _get_valid_python_assertions(self, docs: list[str]) -> list[str]:
        for i, doc in enumerate(docs):
            if not self._is_valid_python_code(doc):
                docs.pop(i)
        return docs


class CTRFunctionSignature:
    def __init__(self, func: Callable):
        self.func = func
        self.signature = inspect.signature(func)
        self.parameters = self.signature.parameters
        self.return_annotation = self.signature.return_annotation

    def get_params_as_function_signature(self):
        return ", ".join(self.func.__code__.co_varnames)

    def raise_for_missing_annotation(self):
        for name, param in self.parameters.items():
            if param.annotation == inspect._empty:
                raise MissingAnnotationError(
                    f"Missing annotation for parameter `{name}`"
                )


class Contract:
    POST_CONDITION_RETURN = "__return__"
    CTR_BASE_FUNCTION_NAME = "__ctr_"

    def __init__(
        self,
        pre: Optional[List[str]]  = None,
        post: Optional[List[str]] = None,
        *,
        check_type: bool = True,
        strict: bool = True,
    ):
        self.pre_contracts = pre or []
        self.post_contracts = post or []

        self.check_type = check_type
        self.strict = strict

        self.namespace = {}

    @staticmethod
    def _pass_method(*args, **kwargs):
        pass

    def generate_pre_assertion_function(
        self,
        func: Callable,
        pre_assertion_function_name: str,
        pre_assertions: list[str],
        func_signature: CTRFunctionSignature,
    ):
        ctr_function_signature = f"def {pre_assertion_function_name}({func_signature.get_params_as_function_signature()}):"
        function = self._get_pre_assertion_function(
            pre_assertions, ctr_function_signature
        )

        exec(function, self.namespace)
        contract_function = self.namespace[pre_assertion_function_name]

        setattr(func, pre_assertion_function_name, contract_function)

    def generate_type_assertion_function(
        self,
        func: Callable,
        contract_type_function_name: str,
        func_signature: CTRFunctionSignature,
    ):
        ctr_function_type_signature = f"def {contract_type_function_name}({func_signature.get_params_as_function_signature()}):"
        ctr_function_type = self._get_dynamic_type_checking_function(
            func_signature, ctr_function_type_signature
        )

        exec(ctr_function_type, self.namespace)
        contract_type_function = self.namespace[contract_type_function_name]
        setattr(func, contract_type_function_name, contract_type_function)

    def generate_post_assertion_function(
        self,
        func: Callable,
        post_assertion_function_name: str,
        post_assertions: list[str],
    ):
        ctr_function_signature = f"def {post_assertion_function_name}(return_value):"
        function = self._get_post_assertion_function(
            post_assertions, ctr_function_signature
        )

        exec(function, self.namespace)
        contract_function = self.namespace[post_assertion_function_name]
        setattr(func, post_assertion_function_name, contract_function)

    def __call__(self, func: Callable):
        docstring_parser = CTRDocstringParser(inspect.getdoc(func))
        func_signature = CTRFunctionSignature(func)

        contract_type_function_name = f"__ctr_type_{func.__name__}"
        pre_assertion_function_name = f"__ctr_pre_{func.__name__}"
        post_assertion_function_name = f"__ctr_post_{func.__name__}"

        pre_assertions = docstring_parser.pre_assertions + self.pre_contracts
        post_assertions = docstring_parser.post_assertions + self.post_contracts

        if self.check_type:
            func_signature.raise_for_missing_annotation()
            self.generate_type_assertion_function(
                func, contract_type_function_name, func_signature
            )

        if pre_assertions:
            self.generate_pre_assertion_function(
                func, pre_assertion_function_name, pre_assertions, func_signature
            )

        if post_assertions:
            self.generate_post_assertion_function(
                func, post_assertion_function_name, post_assertions
            )

        @wraps(func)
        def wrapper(*args, **kwargs):
            if self.check_type:
                try:
                    self._get_method(func, contract_type_function_name)(*args, **kwargs)
                except AssertionError as e:
                    raise ContractTypeError(str(e))
            try:
                self._get_method(func, pre_assertion_function_name)(*args, **kwargs)
            except AssertionError as e:
                raise ContractPreConditionError(str(e))

            return_value = func(*args, **kwargs)

            # TODO
            ## Check the return value type

            try:
                self._get_method(func, post_assertion_function_name)(return_value)
            except AssertionError as e:
                raise ContractPostConditionError(str(e))

            return return_value

        return wrapper

    def _get_method(self, func: Callable, method_name: str):
        return getattr(func, method_name, self._pass_method)

    def _get_pre_assertion_function(
        self, pre_assertions: list[str], function_signature: str
    ):
        return self._get_assertion_function(pre_assertions, function_signature)

    def _get_post_assertion_function(
        self, post_assertions: list[str], function_signature: str
    ):
        return self._get_post_assertion_function(post_assertions, function_signature)

    def _get_assertion_function(self, assertions: list[str], function_signature: str):
        function_lines = [function_signature]
        for rule in assertions:
            assertion_msg = f"Error on rule: `{rule}`"
            function_lines.append(f"""    assert {rule}, {repr(assertion_msg)}""")
        return "\n".join(function_lines)

    def _get_post_assertion_function(
        self, post_assertions: list[str], function_signature: str
    ):
        function_lines = [function_signature]
        for rule in post_assertions:
            assertion_msg = f"Error on rule: `{rule}`"
            rule = rule.replace(self.POST_CONDITION_RETURN, "return_value")
            function_lines.append(f"""    assert {rule}, {repr(assertion_msg)}""")
        return "\n".join(function_lines)

    @staticmethod
    def _get_dynamic_type_checking_function(
        func_signature: CTRFunctionSignature, function_signature: str
    ):
        function_lines = [function_signature]
        for name, param in func_signature.parameters.items():
            annotation_as_str = param.annotation.__name__
            function_lines.append(
                f"""    assert isinstance({name}, {annotation_as_str}), 'Error on rule: "{name} should be of type {annotation_as_str}"'"""
            )
        return "\n".join(function_lines)


class FunctionContract:
    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        def decorator(func: Callable):

            def wrapper(*args, **kwargs):
                pass

            return wrapper
        return decorator

@FunctionContract()
def validate_a(a: int):
    assert a > 10, "a should be greater than 10"


@FunctionContract()
def validate_b(b: int):
    assert b > 10, "a should be greater than 10"


@Validation(b="variable_a")
def toto(a: str, b:str):

    return 1


