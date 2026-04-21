from __future__ import annotations

"""Evaluation helpers for tabular delivery agents."""

from dataclasses import dataclass
from typing import Dict, Iterable, List, Protocol, Sequence

from src.env import GridDeliveryEnv, RewardConfig
from src.instance import GridDeliveryInstance, Position
from src.oracle import shortest_delivery_path


class PolicyLike(Protocol):
    """Minimal policy interface required by the evaluator."""

    def act(self, env: GridDeliveryEnv, *, greedy: bool = True, rng=None) -> int:
        ...


@dataclass(frozen=True)
class RolloutStep:
    """One step from an evaluation rollout."""

    step_index: int
    position: Position
    action: int | None
    reward: float
    done: bool
    success: bool
    reached_goal: bool
    completed_all_deliveries: bool
    invalid_move: bool

    def to_dict(self) -> Dict[str, object]:
        return {
            "step_index": int(self.step_index),
            "position": [int(self.position[0]), int(self.position[1])],
            "action": None if self.action is None else int(self.action),
            "reward": float(self.reward),
            "done": bool(self.done),
            "success": bool(self.success),
            "reached_goal": bool(self.reached_goal),
            "completed_all_deliveries": bool(self.completed_all_deliveries),
            "invalid_move": bool(self.invalid_move),
        }


@dataclass(frozen=True)
class RolloutResult:
    """Single greedy rollout from a policy on one instance."""

    instance_id: str
    total_return: float
    steps: int
    success: bool
    optimal_success: bool
    optimal_steps: int | None
    step_excess: int | None
    reached_goal: bool
    completed_all_deliveries: bool
    invalid_moves: int
    path: List[Position]
    steps_log: List[RolloutStep]

    def to_dict(self) -> Dict[str, object]:
        return {
            "instance_id": self.instance_id,
            "total_return": float(self.total_return),
            "steps": int(self.steps),
            "success": bool(self.success),
            "loose_success": bool(self.success),
            "optimal_success": bool(self.optimal_success),
            "optimal_steps": None if self.optimal_steps is None else int(self.optimal_steps),
            "step_excess": None if self.step_excess is None else int(self.step_excess),
            "reached_goal": bool(self.reached_goal),
            "completed_all_deliveries": bool(self.completed_all_deliveries),
            "invalid_moves": int(self.invalid_moves),
            "path": [[int(row), int(col)] for row, col in self.path],
            "steps_log": [step.to_dict() for step in self.steps_log],
        }


@dataclass(frozen=True)
class EvaluationSummary:
    """Aggregate evaluation metrics over multiple instances."""

    split_name: str
    encoder_name: str
    num_instances: int
    success_rate: float
    loose_success_rate: float
    optimal_success_rate: float
    goal_rate: float
    completion_rate: float
    mean_return: float
    mean_steps: float
    mean_invalid_moves: float
    rollout_summaries: List[RolloutResult]

    def to_dict(self) -> Dict[str, object]:
        return {
            "split_name": self.split_name,
            "encoder_name": self.encoder_name,
            "num_instances": int(self.num_instances),
            "success_rate": float(self.success_rate),
            "loose_success_rate": float(self.loose_success_rate),
            "optimal_success_rate": float(self.optimal_success_rate),
            "goal_rate": float(self.goal_rate),
            "completion_rate": float(self.completion_rate),
            "mean_return": float(self.mean_return),
            "mean_steps": float(self.mean_steps),
            "mean_invalid_moves": float(self.mean_invalid_moves),
            "rollout_summaries": [rollout.to_dict() for rollout in self.rollout_summaries],
        }


def rollout_policy(
    policy: PolicyLike,
    instance: GridDeliveryInstance,
    rewards: RewardConfig | None = None,
) -> RolloutResult:
    """Run one greedy rollout on an instance and keep the full path."""

    oracle = shortest_delivery_path(instance)
    env = GridDeliveryEnv(instance=instance, rewards=rewards)
    env.reset()
    path: List[Position] = [env.position]
    steps_log: List[RolloutStep] = [
        RolloutStep(
            step_index=0,
            position=env.position,
            action=None,
            reward=0.0,
            done=False,
            success=False,
            reached_goal=False,
            completed_all_deliveries=False,
            invalid_move=False,
        )
    ]

    total_return = 0.0
    invalid_moves = 0
    done = False
    final_info: Dict[str, object] = {
        "success": False,
        "reached_goal": False,
        "completed_all_deliveries": False,
    }

    while not done:
        action = policy.act(env, greedy=True)
        _, reward, done, info = env.step(action)
        final_info = info
        total_return += float(reward)
        if bool(info["invalid_move"]):
            invalid_moves += 1
        path.append(env.position)
        steps_log.append(
            RolloutStep(
                step_index=len(steps_log),
                position=env.position,
                action=action,
                reward=float(reward),
                done=bool(done),
                success=bool(info["success"]),
                reached_goal=bool(info["reached_goal"]),
                completed_all_deliveries=bool(info["completed_all_deliveries"]),
                invalid_move=bool(info["invalid_move"]),
            )
        )

    optimal_steps = None if oracle.optimal_steps is None else int(oracle.optimal_steps)
    step_excess = None if optimal_steps is None else int(env.steps - optimal_steps)
    optimal_success = bool(final_info["success"]) and optimal_steps is not None and int(env.steps) == optimal_steps

    return RolloutResult(
        instance_id=instance.instance_id,
        total_return=total_return,
        steps=env.steps,
        success=bool(final_info["success"]),
        optimal_success=optimal_success,
        optimal_steps=optimal_steps,
        step_excess=step_excess,
        reached_goal=bool(final_info["reached_goal"]),
        completed_all_deliveries=bool(final_info["completed_all_deliveries"]),
        invalid_moves=invalid_moves,
        path=path,
        steps_log=steps_log,
    )


def evaluate_policy(
    policy: PolicyLike,
    instances: Sequence[GridDeliveryInstance],
    split_name: str,
    encoder_name: str,
    rewards: RewardConfig | None = None,
) -> EvaluationSummary:
    """Evaluate a policy greedily on a set of instances."""

    rollouts = [rollout_policy(policy, instance, rewards=rewards) for instance in instances]
    num_instances = len(rollouts)
    if num_instances == 0:
        raise ValueError("evaluate_policy requires at least one instance")

    loose_success_rate = sum(1.0 for rollout in rollouts if rollout.success) / num_instances
    optimal_success_rate = sum(1.0 for rollout in rollouts if rollout.optimal_success) / num_instances
    goal_rate = sum(1.0 for rollout in rollouts if rollout.reached_goal) / num_instances
    completion_rate = sum(1.0 for rollout in rollouts if rollout.completed_all_deliveries) / num_instances
    mean_return = sum(rollout.total_return for rollout in rollouts) / num_instances
    mean_steps = sum(float(rollout.steps) for rollout in rollouts) / num_instances
    mean_invalid_moves = sum(float(rollout.invalid_moves) for rollout in rollouts) / num_instances

    return EvaluationSummary(
        split_name=split_name,
        encoder_name=encoder_name,
        num_instances=num_instances,
        success_rate=loose_success_rate,
        loose_success_rate=loose_success_rate,
        optimal_success_rate=optimal_success_rate,
        goal_rate=goal_rate,
        completion_rate=completion_rate,
        mean_return=mean_return,
        mean_steps=mean_steps,
        mean_invalid_moves=mean_invalid_moves,
        rollout_summaries=rollouts,
    )
