"""
Monolithic Federated Learning application.

Everything runs in ONE process: the coordinator, the participants and the
user interface talk through normal method calls (no network, no gRPC).
This is the blueprint for the distributed version.
"""

from dataclasses import dataclass, field
from enum import Enum
import uuid


# ---------- Data structures (later become Protobuf messages) ----------

class JobStatus(Enum):
    CREATED = "CREATED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"


@dataclass
class TrainingConfig:
    num_rounds: int = 1


@dataclass
class TrainingTask:
    job_id: str
    round: int
    global_weight: float


@dataclass
class TrainingResult:
    participant_id: str
    job_id: str
    round: int
    local_weight: float
    num_samples: int


@dataclass
class TrainingJob:
    job_id: str
    config: TrainingConfig
    status: JobStatus = JobStatus.CREATED
    current_round: int = 0
    global_weight: float = 0.0
    results: dict = field(default_factory=dict)  # participant_id -> TrainingResult


# ---------- Participant (later becomes the Node.js process) ----------

class Participant:
    def __init__(self, participant_id, local_data):
        self.participant_id = participant_id
        self.local_data = local_data  # private: never leaves this object

    def train(self, task: TrainingTask) -> TrainingResult:
        # "Local training": the model parameter is the mean of the local data.
        local_weight = sum(self.local_data) / len(self.local_data)
        print(f"  [{self.participant_id}] round {task.round}: "
              f"trained on {len(self.local_data)} samples -> local weight = {local_weight}")
        return TrainingResult(
            participant_id=self.participant_id,
            job_id=task.job_id,
            round=task.round,
            local_weight=local_weight,
            num_samples=len(self.local_data),
        )


# ---------- Coordinator components ----------

class ParticipantRegistry:
    def __init__(self):
        self._participants = {}

    def register(self, participant):
        self._participants[participant.participant_id] = participant

    def get_all(self):
        return list(self._participants.values())


class Aggregator:
    @staticmethod
    def aggregate(results):
        """FedAvg: weighted average by number of samples."""
        total_samples = sum(r.num_samples for r in results)
        return sum(r.local_weight * r.num_samples for r in results) / total_samples


class Coordinator:
    def __init__(self):
        self.registry = ParticipantRegistry()
        self.aggregator = Aggregator()
        self.jobs = {}

    def register_participant(self, participant):
        self.registry.register(participant)
        print(f"[Coordinator] registered participant {participant.participant_id}")

    def create_job(self, config: TrainingConfig) -> TrainingJob:
        job = TrainingJob(job_id="job-" + uuid.uuid4().hex[:6], config=config)
        self.jobs[job.job_id] = job
        print(f"[Coordinator] created {job.job_id} ({config.num_rounds} round(s))")
        return job

    def start_job(self, job_id):
        job = self.jobs[job_id]
        job.status = JobStatus.RUNNING
        job.current_round = 1
        print(f"[Coordinator] started {job_id}")

    def get_task(self, job_id) -> TrainingTask:
        job = self.jobs[job_id]
        return TrainingTask(job.job_id, job.current_round, job.global_weight)

    def submit_result(self, result: TrainingResult):
        job = self.jobs[result.job_id]
        job.results[result.participant_id] = result

        # When every participant has answered, aggregate this round.
        if len(job.results) == len(self.registry.get_all()):
            job.global_weight = self.aggregator.aggregate(list(job.results.values()))
            print(f"[Coordinator] round {job.current_round} aggregated -> "
                  f"global weight = {job.global_weight}")
            job.results = {}
            if job.current_round >= job.config.num_rounds:
                job.status = JobStatus.COMPLETED
            else:
                job.current_round += 1

    def get_status(self, job_id) -> TrainingJob:
        return self.jobs[job_id]

    def run_job(self, job_id):
        """Drive the training loop. In the distributed version, participants
        do this themselves by polling the coordinator over gRPC."""
        job = self.jobs[job_id]
        while job.status == JobStatus.RUNNING:
            for participant in self.registry.get_all():
                task = self.get_task(job_id)              # -> GetTrainingTask
                result = participant.train(task)          # local training
                self.submit_result(result)                # -> SubmitTrainingResult


# ---------- Management interface (later becomes the CLI) ----------

def main():
    coordinator = Coordinator()

    # 1. Register participants (each with its own private data)
    coordinator.register_participant(Participant("A", [8, 10, 12]))
    coordinator.register_participant(Participant("B", [18, 20, 22]))
    coordinator.register_participant(Participant("C", [28, 30, 32]))

    # 2. Create and start a job
    job = coordinator.create_job(TrainingConfig(num_rounds=1))
    coordinator.start_job(job.job_id)

    # 3. Run training
    coordinator.run_job(job.job_id)

    # 4. Show results
    final = coordinator.get_status(job.job_id)
    print(f"\nResult: {final.job_id} | status = {final.status.value} | "
          f"global weight = {final.global_weight}")


if __name__ == "__main__":
    main()