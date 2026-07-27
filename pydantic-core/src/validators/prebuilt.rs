use std::sync::Arc;

use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyType};

use crate::common::prebuilt::get_prebuilt;
use crate::errors::ValResult;
use crate::input::Input;
use crate::tools::SchemaDict;

use super::ValidationState;
use super::{CombinedValidator, SchemaValidator, Validator};

pub struct PrebuiltValidator {
    /// Keeps the referenced `SchemaValidator` alive (and with it, `validator`). This is also the
    /// only field reported to the garbage collector: the contents of `validator` are owned (and
    /// traversed) by the `SchemaValidator`, so they must not be traversed a second time here.
    schema_validator: Py<SchemaValidator>,
    /// The validator to delegate to: either the schema validator's whole tree, or — when the class
    /// has `'after'`/`'wrap'` model validators applied outside of the `model` schema — the inner
    /// `model`/`dataclass` validator.
    validator: Arc<CombinedValidator>,
}

#[allow(clippy::missing_fields_in_debug)] // `schema_validator` is deliberately omitted
impl std::fmt::Debug for PrebuiltValidator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Note: the delegated validator is deliberately not expanded: it is owned by another
        // `SchemaValidator`, and expanding it per reference can result in exponentially large
        // output with highly interconnected models.
        f.debug_struct("PrebuiltValidator")
            .field("validator", &self.validator.get_name())
            .finish()
    }
}

impl PrebuiltValidator {
    pub fn try_get_from_schema(type_: &str, schema: &Bound<'_, PyDict>) -> PyResult<Option<CombinedValidator>> {
        get_prebuilt(type_, schema, "__pydantic_validator__", |py_any| {
            let schema_validator: Py<SchemaValidator> = match py_any.extract() {
                Ok(schema_validator) => schema_validator,
                // If any Pydantic plugin is installed, `__pydantic_validator__` is a
                // `pydantic.plugin._schema_validator.PluggableSchemaValidator` wrapping the actual
                // `SchemaValidator`. Plugin callbacks only fire on the top-level validation entry
                // points (`validate_python()`, etc.) and never on nested validation of a sub-model,
                // so reusing the wrapped validator is behavior-preserving:
                Err(_) => match py_any.getattr(intern!(py_any.py(), "_schema_validator")) {
                    Ok(inner) => match inner.extract() {
                        Ok(schema_validator) => schema_validator,
                        Err(_) => return Ok(None),
                    },
                    Err(_) => return Ok(None),
                },
            };

            // `@model_validator(mode='after')`/`(mode='wrap')` validators are applied *outside* of
            // the `model` (or `dataclass`) schema. A schema referencing this class embeds the
            // class's full core schema, so the function validators are compiled inline before this
            // inner `model`/`dataclass` schema is reached. Delegating to the full prebuilt
            // validator would run the function validators a second time, so strip them and
            // delegate to the inner `model`/`dataclass` validator instead:
            let mut target: &Arc<CombinedValidator> = &schema_validator.get().validator;
            let mut stripped_wrappers = false;
            loop {
                target = match target.as_ref() {
                    CombinedValidator::FunctionAfter(function_validator) => function_validator.inner_validator(),
                    CombinedValidator::FunctionWrap(function_validator) => function_validator.inner_validator(),
                    _ => break,
                };
                stripped_wrappers = true;
            }

            if stripped_wrappers {
                // Only delegate to the stripped validator if it is actually the `model`/`dataclass`
                // validator of the class being referenced. Anything else means the schema was built
                // in some non-standard way; conservatively compile inline instead.
                let class: Bound<'_, PyType> = schema.get_as_req(intern!(schema.py(), "cls"))?;
                let class_matches = match target.as_ref() {
                    CombinedValidator::Model(model_validator) => model_validator.class().is(&class),
                    CombinedValidator::Dataclass(dataclass_validator) => dataclass_validator.class().is(&class),
                    _ => false,
                };
                if !class_matches {
                    return Ok(None);
                }
            }

            let validator = target.clone();
            Ok(Some(
                Self {
                    schema_validator,
                    validator,
                }
                .into(),
            ))
        })
    }
}

impl_py_gc_traverse!(PrebuiltValidator { schema_validator });

impl Validator for PrebuiltValidator {
    fn validate<'py>(
        &self,
        py: Python<'py>,
        input: &(impl Input<'py> + ?Sized),
        state: &mut ValidationState<'_, 'py>,
    ) -> ValResult<Py<PyAny>> {
        self.validator.validate(py, input, state)
    }

    fn get_name(&self) -> &str {
        self.validator.get_name()
    }
}
