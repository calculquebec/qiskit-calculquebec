import numpy as np
import rustworkx as rx

from qiskit.providers import BackendV2, Options
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.transpiler import Target, InstructionProperties
from qiskit.circuit.library import RXGate, RYGate, RZGate, IGate, XGate, YGate, ZGate, CZGate, PhaseGate, TGate, TdgGate, SXGate, SXdgGate, SGate, SdgGate
from qiskit.circuit import Measure, Delay, Parameter, Reset
from qiskit import transpile, QuantumCircuit
from qiskit.visualization import plot_gate_map
import qiskit.qasm2 as qasm2

from qiskit_aer import Aer

from API.adapter import ApiAdapter
from API.client import MonarqClient
from monarq_data import connectivity
from API.job import Job

class monarq_backend(BackendV2):
    def __init__(self):
        
        super().__init__(name="CQ MonarQ")

        self._target = Target(
            "CQ MonarQ", num_qubits=len(connectivity.get("qubits"))
        )

        #TODO prendre les vraies données
        # Generate instruction properties for single qubit gates and a measurement, delay,
        # and reset operation to every qubit in the backend.
        rng = np.random.default_rng(seed=1234567)
        x_props = {}
        z_props = {}
        measure_props = {}
        delay_props = {}
        
        for i in connectivity.get("qubits"):  # Add 1q gates. Globally use virtual rz, x, sx, and measure
            qarg = (i,)
            z_props[qarg] = InstructionProperties(error=0.0, duration=0.0)
            x_props[qarg] = InstructionProperties(
                error=rng.uniform(1e-6, 1e-4),
                duration=rng.uniform(1e-8, 9e-7),
            )
            measure_props[qarg] = InstructionProperties(
                error=rng.uniform(1e-3, 1e-1),
                duration=rng.uniform(1e-8, 9e-7),
            )
            delay_props[qarg] = None
        
        #list of native gates #TODO fix les portes natives +/- _90
        self._target.add_instruction(IGate(), z_props)

        self._target.add_instruction(XGate(), x_props)
        self._target.add_instruction(RXGate(np.pi / 2), x_props)
        # self._target.add_instruction(RXGate(-np.pi / 2), x_props)
        
        self._target.add_instruction(YGate(), x_props)
        self._target.add_instruction(RYGate(np.pi / 2), x_props)
        # self._target.add_instruction(RYGate(-np.pi / 2), x_props)
        
        self._target.add_instruction(ZGate(), z_props)
        # self._target.add_instruction(RZGate(np.pi / 2), z_props)
        # self._target.add_instruction(RZGate(-np.pi / 2), z_props)
        
        self._target.add_instruction(TGate(), z_props)
        self._target.add_instruction(TdgGate(), z_props)
        
        self._target.add_instruction(RZGate(Parameter("theta")), z_props)
        self._target.add_instruction(PhaseGate(Parameter("theta")), z_props)
             
        
        self._target.add_instruction(Measure(), measure_props)
 
        self._target.add_instruction(Delay(Parameter("t")), delay_props)
        # Add chip local 2q gate which is CZ
        cz_props = {}
        for v in connectivity["couplers"].values():
            edge = (v[0], v[1])
            cz_props[edge] = InstructionProperties(
                    error=rng.uniform(7e-4, 5e-3),
                    duration=rng.uniform(1e-8, 9e-7))
                                
        self._target.add_instruction(CZGate(), cz_props)
 
    @property
    def target(self):
        return self._target
 
    @property
    def max_circuits(self):
        return None
 
    @classmethod #TODO utiliser ca au lieu de hardcoder un default dans job.py Job init
    def _default_options(cls):
        return Options(shots=1024)
 
    def run(self, circuit, **kwargs): #TODO merge shots et kwargs pour laisser shots facultatif (avec un default ailleurs)
        if kwargs == {}:#changer/enlever la partie sim (utiliser un autre backend pour la sim?)
            return self._simulate_job(circuit)
        else:
            return self._send_job(circuit, kwargs)
        
        
    def _send_job(self, circuit, kwargs):
        pm = generate_preset_pass_manager(optimization_level=3, backend=self)#TODO explorer autre niveau d'optimization
        transpiled_circuit = pm.run(circuit)
        print(transpiled_circuit)

        ApiAdapter.initialize(MonarqClient(
            host = kwargs.get("host"),
            user=kwargs.get("user"),
            access_token=kwargs.get("access_token"),
            project_name=kwargs.get("project_name")
        ))
        
        result = Job(transpiled_circuit, kwargs.get("shots"), kwargs.get("circuit_name")).run()
        return result


    def _simulate_job(self, circuit): #utile? #TODO? remplacer par un simulateur basé sur monarq
        """Simulate the execution of the job using Qiskit Aer."""
        print("No arguments. Simulating circuit execution...")
        
        # Convert the circuit into a QuantumCircuit object if it is not one already
        if not isinstance(circuit, QuantumCircuit):
            raise ValueError("Provided input is not a valid QuantumCircuit.")
        
        backend = Aer.get_backend('qasm_simulator')
        
        # Execute the circuit on the simulator
        transpiled_circuit = transpile(circuit, backend)
        job=backend.run(transpiled_circuit, shots=1000)
        
        result = job.result()
        #TODO mieux parser les resultats
        return {result.results[0].data}