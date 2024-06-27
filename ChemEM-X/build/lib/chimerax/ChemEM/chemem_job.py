# This file is part of the ChemEM-X software.
#
# Copyright (c) 2024 - Aaron Sweeney
#
# This module was developed by:
#   Aaron Sweeney    <aaron.sweeney AT cssb-hamburg.de>


import subprocess
from chimerax.core.tasks import Job
import json


class RealTimeJob(Job):
    """
    A ChimeraX job that executes a specified shell command and monitors its output in real-time.
    """
    
    def __init__(self, session, command, params, html_view):
        super().__init__(session)
        self.command = command
        self.parameters = params
        self.html_view = html_view
        self.status = "pending"
        self.log_file = []
        
    def run(self):
        """
        Execute the command and monitor its output in real-time.
        """
        self.add_job_to_frontend()
        success = self.execute_command()
        self.on_finish(success)  # Ensure on_finish is called regardless of outcome
        self.terminate()
    
    def add_job_to_frontend(self):
        
        job_data = {
            "id": self.id,
            "status": "running",
        }
        job_data_json = json.dumps(job_data)
        js_code = f"addJob({job_data_json});"
        self.run_js_code(js_code)

    
    def execute_command(self):
        """
        Execute the specified shell command and monitor output in real-time.
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
                    return False
            finally:
                # Properly close stdout and stderr
                self.process.stdout.close()
                self.process.stderr.close()
                # Wait for the process to terminate
                self.process.wait()
                
            if self.process.returncode != 0:
                self.log_file.append(f"Process exited with code {self.process.returncode}\n")
                self.thread_safe_error(f"Process exited with code {self.process.returncode}")
                return False
            return True

        except Exception as e:
            self.log_file.append(f"Exception while executing command: {e}\n")
            self.thread_safe_error(f"Exception while executing command: {e}")
            return False


    def cancel_job(self):
        """
        Cancels the running job if possible.
        """
        print('CANCELING JOB')
        if self.process and self.process.poll() is None:
            self.process.terminate()  # Sends SIGTERM
            self.process.wait()  # Wait for the process to terminate
            #js_code = f'updateJobStatus({self.id}, "Cancelled");'
            #self.run_js_code(js_code)
    
    def run_js_code(self, js_code):
        self.html_view.page().runJavaScript(js_code)
    
    def on_finish(self, success):
        """
        Handles completion of the job, updating the front-end based on whether the job
        was successful or not.
        """
        if success:
            #js_code = "alert('Job completed successfully.');"
            js_code = f'updateJobStatus( {self.id}, "Completed");'
        
        else:
            #js_code = "alert('Job failed.');"
            js_code = f'updateJobStatus( {self.id}, "Failed");'
        
        self.run_js_code(js_code)
        #TODO! write log file
        #log_file_name = os.path.join()
        
    def __str__(self):
        return f"RealTimeJob, ID {self.id}, Command: {self.command}"


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
            return self.jobs[job_id].state
        return None

