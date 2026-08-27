"""Normalized D/N/V tree generation.

The runner does not judge trajectory states, verify factuality, or compute
statistics. Follow-up generation is never conditioned on a trajectory label.
"""

from __future__ import annotations

from itertools import product

from libs.types import SamplerBase

from .interventions import generate_intervention
from .models import complete, sampler_metadata
from .schemas import Action, BranchNode, Message, VerifiedSeed, make_node_id, parent_node_id
from .storage import (
    RunStore,
    conversation_for,
    conversation_for_seed,
    messages_as_dicts,
)


def planned_paths(actions: tuple[str, ...] | list[str], depth: int) -> list[tuple[str, ...]]:
    paths: list[tuple[str, ...]] = []
    for level in range(1, depth + 1):
        paths.extend(product(actions, repeat=level))
    return paths


def expected_node_count(n_actions: int, depth: int) -> int:
    total = 0
    branching = 1
    for _ in range(depth):
        branching *= n_actions
        total += branching
    return total


async def generate_tree(
    store: RunStore,
    seeds: list[VerifiedSeed],
    *,
    answer_model: SamplerBase,
    writer: SamplerBase,
    actions: tuple[str, ...] = ("D", "N", "V"),
    depth: int = 2,
) -> list[BranchNode]:
    """Generate a judge-independent D/N/V tree.

    Resume is by node_id. T1 siblings are generated once and reused.
    """
    store.ensure()
    done = store.completed_ids(store.nodes_path, "node_id")
    nodes_by_id = store.nodes_by_id()
    created: list[BranchNode] = []
    action_enums = [Action(item) for item in actions]

    for seed in seeds:
        if seed.verification_status.value != "VERIFIED_FALSE":
            raise ValueError(f"Refusing to branch non-verified-false seed {seed.seed_id}")
        root = conversation_for_seed(seed)
        frontier: dict[tuple[str, ...], list[Message]] = {(): root}
        for level in range(1, depth + 1):
            parents = [path for path in frontier if len(path) == level - 1]
            next_frontier: dict[tuple[str, ...], list[Message]] = dict(frontier)
            for parent_path in parents:
                parent_messages = frontier[parent_path]
                for action in action_enums:
                    path = parent_path + (action.value,)
                    node_id = make_node_id(seed.seed_id, path)
                    if node_id in done:
                        node = nodes_by_id[node_id]
                        next_frontier[path] = conversation_for(node_id, seed, nodes_by_id)
                        continue
                    intervention = await generate_intervention(
                        action=action,
                        tracked_claim=seed.tracked_claim,
                        conversation=parent_messages,
                        writer=writer,
                    )
                    prompt_messages = messages_as_dicts(parent_messages) + [
                        {"role": "user", "content": intervention.text}
                    ]
                    response = await complete(answer_model, prompt_messages)
                    node = BranchNode(
                        node_id=node_id,
                        seed_id=seed.seed_id,
                        parent_node_id=parent_node_id(seed.seed_id, path),
                        depth=level,
                        path=list(path),
                        action=action,
                        user_message=intervention.text,
                        assistant_response=response.response_text,
                        intervention_metadata=intervention.to_dict(),
                        answer_model_metadata=sampler_metadata(answer_model, response),
                    )
                    store.append_node(node)
                    done.add(node_id)
                    nodes_by_id[node_id] = node
                    created.append(node)
                    next_frontier[path] = parent_messages + [
                        Message(role="user", content=node.user_message),
                        Message(role="assistant", content=node.assistant_response),
                    ]
            frontier = next_frontier
    return created
