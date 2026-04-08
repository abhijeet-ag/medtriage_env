"""
client.py — Typed async (and sync) client for MedTriageEnv.

Async usage:
    async with MedTriageEnv() as env:
        obs = await env.reset(difficulty="hard")
        obs = await env.ask("Do you have chest pain?")
        obs = await env.order_test("ECG")
        obs = await env.submit("STEMI", triage_level="immediate")

Sync usage:
    with MedTriageEnv().sync() as env:
        obs = env.reset(difficulty="easy")
        obs = env.ask("Any nausea or vomiting?")
        obs = env.submit("acute_appendicitis", triage_level="urgent")
"""

from __future__ import annotations

from openenv.core.env_client import EnvClient

from medtriage_env.models import MedTriageAction, MedTriageObservation, MedTriageState


class MedTriageEnv(EnvClient):
    """Async client for the MedTriageEnv OpenEnv environment.

    Inherits connection management, ``reset()``, and ``step()`` from
    ``EnvClient``.  The three helper methods below map clinical operations
    onto typed ``MedTriageAction`` objects so callers never construct
    action dicts by hand.
    """

    action_type      = MedTriageAction
    observation_type = MedTriageObservation

    # ------------------------------------------------------------------
    # Async helpers
    # ------------------------------------------------------------------

    async def ask(self, question: str) -> MedTriageObservation:
        """Pose a natural-language question to the patient.

        Args:
            question: Free-text question string shown to the simulated patient.

        Returns:
            Observation containing the patient's response.
        """
        return await self.step(
            MedTriageAction(action_type="ask_question", question=question)
        )

    async def order_test(self, test_name: str) -> MedTriageObservation:
        """Order a diagnostic investigation.

        Args:
            test_name: Name of the test to order (e.g. ``"ECG"``, ``"CBC"``).

        Returns:
            Observation containing the test result.
        """
        return await self.step(
            MedTriageAction(action_type="order_test", test_name=test_name)
        )

    async def submit(self, diagnosis: str, triage_level: str) -> MedTriageObservation:
        """Submit a final diagnosis and triage level to close the episode.

        Args:
            diagnosis:    Clinical diagnosis string (e.g. ``"acute_appendicitis"``).
            triage_level: One of ``"immediate"``, ``"urgent"``,
                          ``"semi-urgent"``, ``"non-urgent"``.

        Returns:
            Terminal observation containing the episode score and breakdown.
        """
        return await self.step(
            MedTriageAction(
                action_type="submit_diagnosis",
                diagnosis=diagnosis,
                triage_level=triage_level,
            )
        )


    # ------------------------------------------------------------------
    # Required abstract method implementations
    # ------------------------------------------------------------------

    def _step_payload(self, action: MedTriageAction) -> dict:
        """Convert MedTriageAction to JSON payload for the server."""
        return action.model_dump(exclude_none=True)

    def _parse_result(self, payload: dict) -> "StepResult[MedTriageObservation]":
        """Convert server JSON response to StepResult[MedTriageObservation]."""
        from openenv.core.env_client import StepResult
        obs_data = payload.get("observation", payload)
        obs = MedTriageObservation(**obs_data) if obs_data else MedTriageObservation(
            observation_type="error",
            content="Empty response",
            available_actions=[],
        )
        done = obs_data.get("episode_done", False) or payload.get("done", False)
        return StepResult(
            observation=obs,
            reward=payload.get("reward", obs_data.get("step_reward", 0.0)),
            done=done,
        )

    def _parse_state(self, payload: dict) -> MedTriageState:
        """Convert server JSON response to MedTriageState."""
        return MedTriageState(**payload) if payload else MedTriageState(
            episode_id="unknown",
            case_id="unknown",
            difficulty="unknown",
        )

    # ------------------------------------------------------------------
    # Sync wrappers
    # ------------------------------------------------------------------

    def sync(self) -> EnvClient.SyncWrapper:
        """Return a synchronous wrapper exposing the same interface.

        The wrapper proxies every async method (``reset``, ``ask``,
        ``order_test``, ``submit``, ``step``) as a blocking call, suitable
        for scripts and notebooks that do not run an asyncio event loop.

        Returns:
            ``EnvClient.SyncWrapper`` instance bound to this client.
        """
        return super().sync()
