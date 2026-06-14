"""Lineage traversal — ancestor and descendant trees.

Both walks include a cycle guard. Genealogical data can legitimately contain
cycles (e.g. cousin marriages, or pedigree collapse on royal lines), and bad
data definitely can. When a previously-visited individual is encountered,
they appear as a leaf in the tree — we record the identity but do not recurse.
"""

from __future__ import annotations

import logging

from ancestors.models import AncestorNode, DescendantNode, GedcomDatabase
from ancestors.tools.gedcom import IndividualNotFoundError

log = logging.getLogger(__name__)


def get_ancestors(
    db: GedcomDatabase, id: str, generations: int
) -> AncestorNode:
    """Build the ancestor tree rooted at `id`, up to `generations` deep.

    generations=0 returns just the root with no parents resolved.
    generations=1 returns root + parents. generations=2 adds grandparents.
    """
    if id not in db.individuals:
        raise IndividualNotFoundError(id)
    if generations < 0:
        raise ValueError(f"generations must be >= 0, got {generations}")

    visited: set[str] = set()
    return _build_ancestor_node(db, id, generation=0, max_gen=generations, visited=visited)


def get_descendants(
    db: GedcomDatabase, id: str, generations: int
) -> DescendantNode:
    """Build the descendant tree rooted at `id`, up to `generations` deep."""
    if id not in db.individuals:
        raise IndividualNotFoundError(id)
    if generations < 0:
        raise ValueError(f"generations must be >= 0, got {generations}")

    visited: set[str] = set()
    return _build_descendant_node(
        db, id, generation=0, max_gen=generations, visited=visited
    )


def _build_ancestor_node(
    db: GedcomDatabase,
    id: str,
    generation: int,
    max_gen: int,
    visited: set[str],
) -> AncestorNode:
    ind = db.individuals[id]
    node = AncestorNode(individual=ind, generation=generation)

    if generation >= max_gen or id in visited:
        return node
    visited.add(id)

    # An individual is "child of" zero or more families (typically one, but
    # adoptions, step-families, and unmerged duplicates can produce more).
    # We resolve parents from the first FAMC; the agent layer can investigate
    # additional FAMC links explicitly if needed.
    if not ind.families_as_child:
        return node
    family = db.families.get(ind.families_as_child[0])
    if family is None:
        return node

    if family.husband_id and family.husband_id in db.individuals:
        node.father = _build_ancestor_node(
            db, family.husband_id, generation + 1, max_gen, visited
        )
    if family.wife_id and family.wife_id in db.individuals:
        node.mother = _build_ancestor_node(
            db, family.wife_id, generation + 1, max_gen, visited
        )
    return node


def _build_descendant_node(
    db: GedcomDatabase,
    id: str,
    generation: int,
    max_gen: int,
    visited: set[str],
) -> DescendantNode:
    ind = db.individuals[id]
    node = DescendantNode(individual=ind, generation=generation)

    if generation >= max_gen or id in visited:
        return node
    visited.add(id)

    seen_children: set[str] = set()
    for fam_id in ind.families_as_spouse:
        family = db.families.get(fam_id)
        if family is None:
            continue
        for child_id in family.children_ids:
            if child_id in seen_children or child_id not in db.individuals:
                continue
            seen_children.add(child_id)
            node.children.append(
                _build_descendant_node(
                    db, child_id, generation + 1, max_gen, visited
                )
            )
    return node


def count_ancestors(node: AncestorNode) -> int:
    """Total individuals in an ancestor tree (root included)."""
    total = 1
    if node.father is not None:
        total += count_ancestors(node.father)
    if node.mother is not None:
        total += count_ancestors(node.mother)
    return total


def count_descendants(node: DescendantNode) -> int:
    """Total individuals in a descendant tree (root included)."""
    return 1 + sum(count_descendants(c) for c in node.children)
