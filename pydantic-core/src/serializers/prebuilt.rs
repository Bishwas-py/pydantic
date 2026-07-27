use std::borrow::Cow;
use std::sync::Arc;

use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyType};

use crate::SchemaSerializer;
use crate::serializers::SerializationState;
use crate::tools::SchemaDict;
use crate::{common::prebuilt::get_prebuilt, serializers::polymorphism_trampoline::PolymorphismTrampoline};

use super::shared::{CombinedSerializer, TypeSerializer};

pub struct PrebuiltSerializer {
    /// Keeps the referenced `SchemaSerializer` alive (and with it, `serializer`). This is also the
    /// only field reported to the garbage collector: the contents of `serializer` are owned (and
    /// traversed) by the `SchemaSerializer`, so they must not be traversed a second time here.
    schema_serializer: Py<SchemaSerializer>,
    /// The serializer to delegate to: either the schema serializer's whole tree, or — when the
    /// class has a `'wrap'` model serializer — the serializer that the function wraps.
    serializer: Arc<CombinedSerializer>,
}

#[allow(clippy::missing_fields_in_debug)] // `schema_serializer` is deliberately omitted
impl std::fmt::Debug for PrebuiltSerializer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Note: the delegated serializer is deliberately not expanded: it is owned by another
        // `SchemaSerializer`, and expanding it per reference can result in exponentially large
        // output with highly interconnected models.
        f.debug_struct("PrebuiltSerializer")
            .field("serializer", &self.serializer.get_name())
            .finish()
    }
}

impl PrebuiltSerializer {
    pub fn try_get_from_schema(type_: &str, schema: &Bound<'_, PyDict>) -> PyResult<Option<CombinedSerializer>> {
        get_prebuilt(type_, schema, "__pydantic_serializer__", |py_any| {
            let schema_serializer = py_any.extract::<Py<SchemaSerializer>>()?;

            let mut target: &Arc<CombinedSerializer> = &schema_serializer.get().serializer;

            // it is very likely that the prebuilt serializer is a polymorphism trampoline, peek
            // through it for the sake of the `function-wrap` handling below
            let mut peeked = target;
            if let CombinedSerializer::PolymorphismTrampoline(PolymorphismTrampoline {
                serializer: inner_serializer,
                ..
            }) = peeked.as_ref()
            {
                peeked = inner_serializer;
            }

            // A `'wrap'` model serializer is applied *around* the `model`/`dataclass` serializer.
            // A schema referencing this class embeds the class's full core schema (including the
            // `serialization` key), so the wrap function serializer is compiled inline before this
            // `model`/`dataclass` schema (with the `serialization` key removed) is reached.
            // Delegating to the full prebuilt serializer would apply the wrap function a second
            // time, so delegate to the serializer that the function wraps instead:
            if let CombinedSerializer::FunctionWrap(function_serializer) = peeked.as_ref() {
                let stripped = function_serializer.inner_serializer();

                // Only delegate to the stripped serializer if it is the polymorphism trampoline
                // around the serializer of the class being referenced. The inner serializer of a
                // `model`/`dataclass` schema is always built wrapped in a trampoline, and
                // delegating to anything else (e.g. a bare model serializer) could skip the
                // polymorphic subclass dispatch that inline compilation would preserve. Anything
                // else also means the schema was built in some non-standard way; conservatively
                // compile inline instead.
                let class: Bound<'_, PyType> = schema.get_as_req(intern!(schema.py(), "cls"))?;
                let class_matches = match stripped.as_ref() {
                    CombinedSerializer::PolymorphismTrampoline(trampoline) => {
                        trampoline.class.bind(schema.py()).is(&class)
                    }
                    _ => false,
                };
                if !class_matches {
                    return Ok(None);
                }

                target = stripped;
            }

            let serializer = target.clone();
            Ok(Some(
                Self {
                    schema_serializer,
                    serializer,
                }
                .into(),
            ))
        })
    }
}

impl_py_gc_traverse!(PrebuiltSerializer { schema_serializer });

impl TypeSerializer for PrebuiltSerializer {
    fn to_python<'py>(&self, value: &Bound<'py, PyAny>, state: &mut SerializationState<'py>) -> PyResult<Py<PyAny>> {
        self.serializer.to_python_no_infer(value, state)
    }

    fn json_key<'a, 'py>(
        &self,
        key: &'a Bound<'py, PyAny>,
        state: &mut SerializationState<'py>,
    ) -> PyResult<Cow<'a, str>> {
        self.serializer.json_key_no_infer(key, state)
    }

    fn serde_serialize<'py, S: serde::ser::Serializer>(
        &self,
        value: &Bound<'py, PyAny>,
        serializer: S,
        state: &mut SerializationState<'py>,
    ) -> Result<S::Ok, S::Error> {
        self.serializer.serde_serialize_no_infer(value, serializer, state)
    }

    fn get_name(&self) -> &str {
        self.serializer.get_name()
    }

    fn retry_with_lax_check(&self) -> bool {
        self.serializer.retry_with_lax_check()
    }
}
