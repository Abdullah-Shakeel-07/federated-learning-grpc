"""
Coordinator Service - the only gRPC server in the system.

Manages participants and training jobs, hands out training tasks,
collects local results and aggregates them with FedAvg.

Run:  python3 coordinator/server.py [--port 50051]
"""

import argparse
import threading
import time
import uuid
from concurrent import futures

import grpc

import fl_pb2
import fl_pb2_grpc


def log(msg):
    print(f"{time.strftime('%H:%M:%S')} [Coordinator] {msg}", flush=True)


class Job:
    """In-memory state of one training job (= TrainingJob in the monolith)."""

    def __init__(self, job_id, num_rounds):
        self.job_id = job_id
        self.num_rounds = num_rounds
        self.status = fl_pb2.CREATED
        self.current_round = 0
        self.global_weight = 0.0
        self.participants = []   # snapshot of registered participants at start
        self.results = {}        # participant_id -> TrainingResult (current round)


class CoordinatorServicer(fl_pb2_grpc.CoordinatorServicer):

    def __init__(self):
        self.participants = []   # registered participant ids
        self.jobs = {}           # job_id -> Job
        self.lock = threading.Lock()  # gRPC calls run in parallel threads

    # ---------- Called by the CLI ----------

    def RegisterParticipant(self, request, context):
        pid = request.participant_id.strip()
        with self.lock:
            if not pid:
                return fl_pb2.RegisterParticipantResponse(success=False, message="Empty participant id")
            if pid in self.participants:
                return fl_pb2.RegisterParticipantResponse(success=False, message=f"{pid} is already registered")
            self.participants.append(pid)
            log(f"RegisterParticipant: '{pid}' registered (total: {len(self.participants)})")
            return fl_pb2.RegisterParticipantResponse(success=True, message=f"{pid} registered")

    def CreateExperiment(self, request, context):
        num_rounds = request.num_rounds if request.num_rounds > 0 else 1
        with self.lock:
            job_id = "job-" + uuid.uuid4().hex[:6]
            self.jobs[job_id] = Job(job_id, num_rounds)
            log(f"CreateExperiment: {job_id} created ({num_rounds} round(s))")
            return fl_pb2.CreateExperimentResponse(job_id=job_id, message=f"{job_id} created")

    def StartTraining(self, request, context):
        with self.lock:
            job = self.jobs.get(request.job_id)
            if job is None:
                return fl_pb2.StartTrainingResponse(success=False, message="Unknown job id")
            if job.status != fl_pb2.CREATED:
                return fl_pb2.StartTrainingResponse(success=False, message="Job already started")
            if not self.participants:
                return fl_pb2.StartTrainingResponse(success=False, message="No participants registered")
            job.participants = list(self.participants)
            job.status = fl_pb2.RUNNING
            job.current_round = 1
            log(f"StartTraining: {job.job_id} started with participants {job.participants}")
            log(f"Round 1/{job.num_rounds}: waiting for participants to fetch their task")
            return fl_pb2.StartTrainingResponse(success=True, message=f"{job.job_id} started")

    def GetTrainingStatus(self, request, context):
        with self.lock:
            job = self.jobs.get(request.job_id)
            if job is None:
                context.abort(grpc.StatusCode.NOT_FOUND, "Unknown job id")
            return fl_pb2.TrainingStatus(
                job_id=job.job_id,
                status=job.status,
                current_round=job.current_round,
                num_rounds=job.num_rounds,
                global_weight=job.global_weight,
                results_received=len(job.results),
                total_participants=len(job.participants),
            )

    # ---------- Called by the Participants ----------

    def GetTrainingTask(self, request, context):
        pid = request.participant_id
        with self.lock:
            for job in self.jobs.values():
                if (job.status == fl_pb2.RUNNING
                        and pid in job.participants
                        and pid not in job.results):
                    log(f"GetTrainingTask: sending task to '{pid}' "
                        f"({job.job_id}, round {job.current_round}, global weight {job.global_weight})")
                    return fl_pb2.TrainingTask(
                        has_task=True,
                        job_id=job.job_id,
                        round=job.current_round,
                        global_weight=job.global_weight,
                    )
        return fl_pb2.TrainingTask(has_task=False)  # nothing to do yet

    def SubmitTrainingResult(self, request, context):
        with self.lock:
            job = self.jobs.get(request.job_id)
            if job is None or job.status != fl_pb2.RUNNING:
                return fl_pb2.SubmitTrainingResultResponse(accepted=False, message="Job is not running")
            if request.participant_id not in job.participants:
                return fl_pb2.SubmitTrainingResultResponse(accepted=False, message="Participant not in this job")
            if request.round != job.current_round:
                return fl_pb2.SubmitTrainingResultResponse(accepted=False, message="Wrong round")

            job.results[request.participant_id] = request
            log(f"SubmitTrainingResult: '{request.participant_id}' -> local weight {request.local_weight} "
                f"({request.num_samples} samples) [{len(job.results)}/{len(job.participants)}]")

            if len(job.results) == len(job.participants):
                self._aggregate(job)
            return fl_pb2.SubmitTrainingResultResponse(accepted=True, message="Result accepted")

    # ---------- Aggregation (FedAvg) ----------

    def _aggregate(self, job):
        results = list(job.results.values())
        total = sum(r.num_samples for r in results)
        job.global_weight = sum(r.local_weight * r.num_samples for r in results) / total
        parts = " + ".join(f"{r.local_weight}x{r.num_samples}" for r in results)
        log(f"Aggregating round {job.current_round}: ({parts}) / {total} = {job.global_weight}")
        job.results = {}
        if job.current_round >= job.num_rounds:
            job.status = fl_pb2.COMPLETED
            log(f"{job.job_id} COMPLETED - final global weight = {job.global_weight}")
        else:
            job.current_round += 1
            log(f"Round {job.current_round}/{job.num_rounds}: waiting for participants")


def serve(port):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    fl_pb2_grpc.add_CoordinatorServicer_to_server(CoordinatorServicer(), server)
    server.add_insecure_port(f"0.0.0.0:{port}")
    server.start()
    log(f"gRPC server listening on port {port}")
    server.wait_for_termination()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FL Coordinator Service")
    parser.add_argument("--port", type=int, default=50051)
    serve(parser.parse_args().port)
