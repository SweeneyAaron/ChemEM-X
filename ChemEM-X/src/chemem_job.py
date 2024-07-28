# This file is part of the ChemEM-X software.
#
# Copyright (c) 2024 - Aaron Sweeney
#
# This module was developed by:
#   Aaron Sweeney    <aaron.sweeney AT cssb-hamburg.de>


import subprocess
from chimerax.core.tasks import Job, Task, TaskState
import json
import datetime
import numpy as np

CHEMEM_JOB = 'chemem'
EXPORT_SIMULATION = 'chemem.export_simulation'
SIMULATION_JOB = 'chemem-X simulation'

class SimulationJob(Task):
    def __init__(self, session, chemem, job_type = SIMULATION_JOB):
        super().__init__(session)
        self.chemem = chemem 
        self.job_type = job_type
        self.running = True
        self._pause = False
        self.started = False
        #set these to set a new tempreture
        self._set_temp = None
        
        self.tug = None
        self.hbond_tug = None
        #[15879, np.array([128.76779874, 130.55397987, 173.85587692])]
        self.step_size = 5
        
    def terminate(self):
        """Terminate this task.

        This method should be overridden to clean up
        task data structures.  This base method should be
        called as the last step of task deletion.

        """
        self.session.tasks.remove(self)
        self.end_time = datetime.datetime.now()
        
        if self._terminate is not None:
            self._terminate.set()
            
        self.state = TaskState.TERMINATING
    
    
    def run(self, *args, **kw):
        
        self.started = True
        self.chemem.simulation.minimise_system()
        self.chemem.update_simulation_model()
        self.step_count = 0
        while self.running :
            #For debugging!!
            self.step_count += self.step_size
            if not self._pause:
                self.chemem.simulation.step(self.step_size)
                self.chemem.update_simulation_model()
            
            #can update temp while paused
            if self._set_temp is not None:
                self.chemem.simulation.set_tempreture(self._set_temp)
                self.chemem.update_simulation_model()
                self._set_temp = None
            
            
            if self.tug is not None:
                
                if self.tug.atom_idx is not None:
                    
                    self.chemem.simulation.update_tug_force_for_atom(self.tug.atom_idx, self.tug.end_coord)
                    for num in range(20):
                        
                        self.chemem.simulation.step(self.step_size)
                        self.chemem.update_simulation_model()
                    self.chemem.simulation.update_tug_force_for_atom(self.tug.atom_idx, self.tug.end_coord, tug_k = 0.0)
                    self.tug.atom_idx = None
                        #set the tug stuff back to None
            
            if self.hbond_tug is not None:
                
                self.chemem.simulation.update_hbond_tug_force_for_atom(self.hbond_tug[0],
                                                                       hbond_dist_k = self.hbond_tug[1],
                                                                       hbond_angle_k = self.hbond_tug[2])
                for num in range(20):
                    #run for a while incase the simulation is paused
                    self.chemem.simulation.step(self.step_size)
                    self.chemem.update_simulation_model()
                self.hbond_tug = None
            
            #if self.step_count == 250:
            #    print('HBOND TUG TEST TIME')
            #    self.chemem.simulation.update_hbond_tug_force_for_atom(self.hbond_tug)
                
            
                
            
            
    
    
    
    def on_finish(self):
        """Callback method executed after task thread terminates.

        This callback is executed in the UI thread after the
        :py:meth:`run` method returns.  By default, it does nothing.

        """
        self.terminate()
            
            

class LocalChemEMJob(Task):
    
    def __init__(self, session, command, params, job_type):
        super().__init__(session)
        self.command = command 
        self.params = params
        self.log_file = []
        self.job_type =  job_type
    #possible you need to add terminate to cleanup threads?
    #self._thread.join() wait for the thread to finish??
    
    def terminate(self):
        """Terminate this task.

        This method should be overridden to clean up
        task data structures.  This base method should be
        called as the last step of task deletion.

        """
        self.session.tasks.remove(self)
        self.end_time = datetime.datetime.now()
        
        if self._terminate is not None:
            self._terminate.set()
            
        self.state = TaskState.TERMINATING
    
    def run(self, *args, **kw):
        """Run the task.

        This method must be overridden to implement actual functionality.
        :py:meth:`terminating` should be checked regularly to see whether
        user has requested termination.

        """
        try:
            self.process = subprocess.Popen(self.command, shell=True, text=True,
                                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            try:
                for line in self.process.stdout:
                    self.log_file.append(f"Output: {line.strip()}\n")
                    self.thread_safe_log(f"Output: {line.strip()}")

                err = self.process.stderr.read()
                if err:
                    self.log_file.append(f"Error: {err.strip()}\n")
                    self.thread_safe_error(f"Error: {err.strip()}")
                    self.success = False
                    return 
            finally:
                # Properly close stdout and stderr
                self.process.stdout.close()
                self.process.stderr.close()
                # Wait for the process to terminate
                self.process.wait()
                
            if self.process.returncode != 0:
                self.log_file.append(f"Process exited with code {self.process.returncode}\n")
                self.thread_safe_error(f"Process exited with code {self.process.returncode}")
                self.success = False
                return 
            self.success = True
            return 

        except Exception as e:
            self.log_file.append(f"Exception while executing command: {e}\n")
            self.thread_safe_error(f"Exception while executing command: {e}")
            self.success = False
            return 
    
    def on_finish(self):
        """Callback method executed after task thread terminates.

        This callback is executed in the UI thread after the
        :py:meth:`run` method returns.  By default, it does nothing.

        """
        self.terminate()
        
    
    def cancel_job(self):
        """
        Cancels the running job if possible.
        """
        
        if self.process and self.process.poll() is None:
            self.process.terminate()  # Sends SIGTERM
            self.process.wait()
            self.terminate()
    
    

class JobHandler:
    def __init__(self):
        
        self.jobs = {}  # Dictionary to store jobs with a unique identifier as the key
    
    
    def add_job(self, job):
        """ Adds a job to the handler. """
        if job.id in self.jobs:
            raise ValueError("A job with this ID already exists.")
        self.jobs[job.id] = job

    def remove_job(self, job_id):
        """ Removes a job from the handler. """
        if job_id in self.jobs:
            self.stop_job(job_id)
            del self.jobs[job_id]


    def stop_job(self, job_id):
        """ Stops a specific job by ID. """
        if job_id in self.jobs:
            self.jobs[job_id].cancel_job()
           

    def get_job_status(self, job_id):
        """ Returns the status of a specific job. """
        if job_id in self.jobs:
            return self.jobs[job_id].status
        return None

