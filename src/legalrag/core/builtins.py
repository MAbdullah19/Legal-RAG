"""Import side effects register all built-in components.

Importing this module populates the registry with the components that ship in
core. The pipeline runner imports it, so any experiment config can name these
without the caller worrying about import order. New builtin components add an
import line here; jurisdiction packs register their own on import of the pack.
"""

from __future__ import annotations

from legalrag.core.generate import assemblers as _assemblers  # noqa: F401
from legalrag.core.generate import generators as _generators  # noqa: F401
from legalrag.core.generate import generators_external as _generators_external  # noqa: F401
from legalrag.core.index import chunkers as _chunkers  # noqa: F401
from legalrag.core.index import embedders as _embedders  # noqa: F401
from legalrag.core.index import embedders_external as _embedders_external  # noqa: F401
from legalrag.core.retrieve import fusion as _fusion  # noqa: F401
from legalrag.core.retrieve import graph as _graph_expand  # noqa: F401
from legalrag.core.retrieve import lexical as _lexical  # noqa: F401
from legalrag.core.retrieve import rerankers as _rerankers  # noqa: F401
from legalrag.core.retrieve import rerankers_external as _rerankers_external  # noqa: F401
from legalrag.core.retrieve import transformers as _transformers  # noqa: F401
from legalrag.core.retrieve import vector as _vector  # noqa: F401
from legalrag.core.verify import checks as _checks  # noqa: F401
