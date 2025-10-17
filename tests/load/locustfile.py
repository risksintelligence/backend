"""
Locust load testing configuration for RiskX API
"""

from locust import HttpUser, task, between
import random
import json


class RiskXAPIUser(HttpUser):
    """Simulates a user of the RiskX API"""
    
    wait_time = between(1, 5)  # Wait 1-5 seconds between requests
    
    def on_start(self):
        """Called when a user starts"""
        # Check if the API is accessible
        response = self.client.get("/api/v1/health")
        if response.status_code != 200:
            print(f"API health check failed: {response.status_code}")
    
    @task(3)
    def get_health_status(self):
        """Test health endpoint (most frequent)"""
        with self.client.get("/api/v1/health", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Health check failed: {response.status_code}")
    
    @task(2)
    def get_risk_score(self):
        """Test risk score endpoint"""
        with self.client.get("/api/v1/risk/score", catch_response=True) as response:
            if response.status_code == 200:
                try:
                    data = response.json()
                    if "overall_risk_score" in data:
                        response.success()
                    else:
                        response.failure("Missing overall_risk_score in response")
                except json.JSONDecodeError:
                    response.failure("Invalid JSON response")
            elif response.status_code == 503:
                # Service unavailable is acceptable during heavy load
                response.success()
            else:
                response.failure(f"Risk score failed: {response.status_code}")
    
    @task(2)
    def get_risk_factors(self):
        """Test risk factors endpoint"""
        with self.client.get("/api/v1/risk/factors", catch_response=True) as response:
            if response.status_code == 200:
                try:
                    data = response.json()
                    if "economic_indicators" in data or "financial_indicators" in data:
                        response.success()
                    else:
                        response.failure("Missing expected data in response")
                except json.JSONDecodeError:
                    response.failure("Invalid JSON response")
            elif response.status_code == 503:
                response.success()
            else:
                response.failure(f"Risk factors failed: {response.status_code}")
    
    @task(1)
    def get_risk_categories(self):
        """Test risk categories endpoint"""
        with self.client.get("/api/v1/risk/categories", catch_response=True) as response:
            if response.status_code == 200:
                try:
                    data = response.json()
                    if "categories" in data:
                        response.success()
                    else:
                        response.failure("Missing categories in response")
                except json.JSONDecodeError:
                    response.failure("Invalid JSON response")
            else:
                response.failure(f"Risk categories failed: {response.status_code}")
    
    @task(1)
    def get_analytics_aggregation(self):
        """Test analytics aggregation endpoint"""
        # Add random query parameters to test different scenarios
        params = {}
        if random.choice([True, False]):
            params['limit'] = random.randint(10, 100)
        
        with self.client.get("/api/v1/analytics/aggregation", params=params, catch_response=True) as response:
            if response.status_code == 200:
                try:
                    data = response.json()
                    response.success()
                except json.JSONDecodeError:
                    response.failure("Invalid JSON response")
            elif response.status_code == 503:
                response.success()
            else:
                response.failure(f"Analytics aggregation failed: {response.status_code}")
    
    @task(1)
    def get_analytics_health(self):
        """Test analytics health endpoint"""
        with self.client.get("/api/v1/analytics/health", catch_response=True) as response:
            if response.status_code == 200:
                try:
                    data = response.json()
                    response.success()
                except json.JSONDecodeError:
                    response.failure("Invalid JSON response")
            else:
                response.failure(f"Analytics health failed: {response.status_code}")
    
    @task(1)
    def get_network_analysis(self):
        """Test network analysis endpoint"""
        with self.client.get("/api/v1/network/analysis", catch_response=True) as response:
            if response.status_code == 200:
                try:
                    data = response.json()
                    if "nodes" in data and "edges" in data:
                        response.success()
                    else:
                        response.failure("Missing network data in response")
                except json.JSONDecodeError:
                    response.failure("Invalid JSON response")
            elif response.status_code == 503:
                response.success()
            else:
                response.failure(f"Network analysis failed: {response.status_code}")


class HeavyAPIUser(HttpUser):
    """Simulates heavy API usage patterns"""
    
    wait_time = between(0.5, 2)  # Faster requests
    
    @task(5)
    def rapid_risk_score_requests(self):
        """Make rapid risk score requests"""
        endpoints = [
            "/api/v1/risk/score",
            "/api/v1/risk/factors", 
            "/api/v1/risk/categories"
        ]
        
        endpoint = random.choice(endpoints)
        with self.client.get(endpoint, catch_response=True) as response:
            if response.status_code in [200, 503]:
                response.success()
            else:
                response.failure(f"Request failed: {response.status_code}")
    
    @task(1)
    def concurrent_analytics_requests(self):
        """Test analytics endpoints under load"""
        with self.client.get("/api/v1/analytics/aggregation", catch_response=True) as response:
            if response.status_code in [200, 503]:
                response.success()
            else:
                response.failure(f"Analytics failed: {response.status_code}")


class RealisticUserWorkflow(HttpUser):
    """Simulates realistic user workflow patterns"""
    
    wait_time = between(2, 10)  # More realistic user thinking time
    
    def on_start(self):
        """User starts by checking system health"""
        self.client.get("/api/v1/health")
    
    @task
    def typical_user_workflow(self):
        """Simulate a typical user workflow"""
        # 1. Check current risk score
        risk_response = self.client.get("/api/v1/risk/score")
        
        if risk_response.status_code == 200:
            # 2. If risk is concerning, get detailed factors
            try:
                risk_data = risk_response.json()
                risk_score = risk_data.get("overall_risk_score", 0)
                
                if risk_score > 0.6:  # High risk threshold
                    # Get detailed risk factors
                    self.client.get("/api/v1/risk/factors")
                    
                    # Check network analysis for interconnections
                    self.client.get("/api/v1/network/analysis")
                
                # 3. Check data health periodically
                if random.random() < 0.3:  # 30% chance
                    self.client.get("/api/v1/analytics/health")
                    
            except (json.JSONDecodeError, KeyError):
                # Handle invalid response gracefully
                pass


class DataAnalystUser(HttpUser):
    """Simulates data analyst usage patterns"""
    
    wait_time = between(3, 15)  # Analysts spend more time reviewing data
    
    @task(2)
    def comprehensive_analysis(self):
        """Perform comprehensive risk analysis"""
        # Get all risk-related data
        endpoints = [
            "/api/v1/risk/score",
            "/api/v1/risk/factors",
            "/api/v1/risk/categories",
            "/api/v1/analytics/aggregation",
            "/api/v1/network/analysis"
        ]
        
        for endpoint in endpoints:
            response = self.client.get(endpoint)
            # Simulate analysis time
            self.wait()
    
    @task(1)
    def periodic_health_check(self):
        """Check system and data health"""
        self.client.get("/api/v1/health")
        self.client.get("/api/v1/analytics/health")


# Load testing scenarios
class QuickLoadTest(HttpUser):
    """Quick load test for basic functionality"""
    wait_time = between(1, 3)
    tasks = [RiskXAPIUser]


class SustainedLoadTest(HttpUser):
    """Sustained load test for endurance testing"""
    wait_time = between(1, 5)
    tasks = [RiskXAPIUser, RealisticUserWorkflow]


class StressTest(HttpUser):
    """Stress test with heavy load"""
    wait_time = between(0.1, 1)
    tasks = [HeavyAPIUser]


class MixedWorkloadTest(HttpUser):
    """Mixed workload simulating different user types"""
    wait_time = between(1, 8)
    tasks = [RiskXAPIUser, RealisticUserWorkflow, DataAnalystUser]


# Custom Locust events for monitoring
from locust import events

@events.request_success.add_listener
def on_request_success(request_type, name, response_time, response_length, **kwargs):
    """Log successful requests for monitoring"""
    if response_time > 5000:  # Log slow requests (>5 seconds)
        print(f"Slow request: {name} took {response_time}ms")

@events.request_failure.add_listener
def on_request_failure(request_type, name, response_time, response_length, exception, **kwargs):
    """Log failed requests for debugging"""
    print(f"Request failed: {name} - {exception}")

@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """Called when test starts"""
    print("RiskX load test starting...")

@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Called when test stops"""
    print("RiskX load test completed")
    
    # Print summary statistics
    stats = environment.stats
    print(f"Total requests: {stats.total.num_requests}")
    print(f"Total failures: {stats.total.num_failures}")
    print(f"Average response time: {stats.total.avg_response_time:.2f}ms")
    print(f"Max response time: {stats.total.max_response_time}ms")
    print(f"Requests per second: {stats.total.current_rps:.2f}")
    
    if stats.total.num_failures > 0:
        failure_rate = (stats.total.num_failures / stats.total.num_requests) * 100
        print(f"Failure rate: {failure_rate:.2f}%")


# Configuration for different test scenarios
LOAD_TEST_CONFIGS = {
    "smoke": {
        "users": 5,
        "spawn_rate": 1,
        "run_time": "2m",
        "user_class": QuickLoadTest
    },
    "load": {
        "users": 25,
        "spawn_rate": 2,
        "run_time": "10m", 
        "user_class": SustainedLoadTest
    },
    "stress": {
        "users": 100,
        "spawn_rate": 10,
        "run_time": "5m",
        "user_class": StressTest
    },
    "spike": {
        "users": 200,
        "spawn_rate": 50,
        "run_time": "3m",
        "user_class": StressTest
    },
    "endurance": {
        "users": 50,
        "spawn_rate": 2,
        "run_time": "30m",
        "user_class": MixedWorkloadTest
    }
}