#!/usr/bin/env python3
"""
Test API endpoints and performance.

This script tests the FastAPI endpoints and performs performance testing.
"""

import os
import sys
import json
import time
import argparse
import requests
import numpy as np
from typing import List, Dict, Any
import structlog

# Configure logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()


class APITester:
    """Test API endpoints and performance."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        Initialize API tester.
        
        Args:
            base_url: Base URL of the API
        """
        self.base_url = base_url
        self.session = requests.Session()
    
    def test_health_endpoint(self) -> Dict[str, Any]:
        """Test health check endpoint."""
        try:
            response = self.session.get(f"{self.base_url}/health")
            response.raise_for_status()
            
            data = response.json()
            logger.info("Health check passed", status=data.get('status'))
            
            return {
                'success': True,
                'status': data.get('status'),
                'model_loaded': data.get('model_loaded'),
                'response_time': response.elapsed.total_seconds()
            }
            
        except Exception as e:
            logger.error("Health check failed", error=str(e))
            return {'success': False, 'error': str(e)}
    
    def test_model_info(self) -> Dict[str, Any]:
        """Test model info endpoint."""
        try:
            response = self.session.get(f"{self.base_url}/model/info")
            response.raise_for_status()
            
            data = response.json()
            logger.info("Model info retrieved", model_name=data.get('model_name'))
            
            return {
                'success': True,
                'model_info': data,
                'response_time': response.elapsed.total_seconds()
            }
            
        except Exception as e:
            logger.error("Model info test failed", error=str(e))
            return {'success': False, 'error': str(e)}
    
    def generate_test_image(self) -> List[List[float]]:
        """Generate a test MNIST image."""
        # Create a simple test image (28x28)
        image = np.random.rand(28, 28).tolist()
        return image
    
    def test_prediction_endpoint(self, image: List[List[float]] = None) -> Dict[str, Any]:
        """Test prediction endpoint."""
        if image is None:
            image = self.generate_test_image()
        
        try:
            payload = {
                "image": image,
                "confidence_threshold": 0.5
            }
            
            response = self.session.post(f"{self.base_url}/predict", json=payload)
            response.raise_for_status()
            
            data = response.json()
            logger.info("Prediction successful", 
                       prediction=data.get('prediction'),
                       confidence=data.get('confidence'))
            
            return {
                'success': True,
                'prediction': data.get('prediction'),
                'confidence': data.get('confidence'),
                'processing_time': data.get('processing_time'),
                'response_time': response.elapsed.total_seconds()
            }
            
        except Exception as e:
            logger.error("Prediction test failed", error=str(e))
            return {'success': False, 'error': str(e)}
    
    def test_batch_prediction(self, num_images: int = 5) -> Dict[str, Any]:
        """Test batch prediction endpoint."""
        try:
            # Generate multiple test images
            images = [self.generate_test_image() for _ in range(num_images)]
            payload = [{"image": img, "confidence_threshold": 0.5} for img in images]
            
            response = self.session.post(f"{self.base_url}/predict/batch", json=payload)
            response.raise_for_status()
            
            data = response.json()
            logger.info("Batch prediction successful", 
                       batch_size=data.get('batch_size'),
                       processing_time=data.get('processing_time'))
            
            return {
                'success': True,
                'batch_size': data.get('batch_size'),
                'processing_time': data.get('processing_time'),
                'response_time': response.elapsed.total_seconds()
            }
            
        except Exception as e:
            logger.error("Batch prediction test failed", error=str(e))
            return {'success': False, 'error': str(e)}
    
    def test_metrics_endpoint(self) -> Dict[str, Any]:
        """Test metrics endpoint."""
        try:
            response = self.session.get(f"{self.base_url}/metrics")
            response.raise_for_status()
            
            logger.info("Metrics endpoint accessible")
            
            return {
                'success': True,
                'response_time': response.elapsed.total_seconds()
            }
            
        except Exception as e:
            logger.error("Metrics test failed", error=str(e))
            return {'success': False, 'error': str(e)}
    
    def performance_test(self, num_requests: int = 100) -> Dict[str, Any]:
        """Run performance test with multiple requests."""
        logger.info("Starting performance test", num_requests=num_requests)
        
        response_times = []
        success_count = 0
        
        for i in range(num_requests):
            start_time = time.time()
            result = self.test_prediction_endpoint()
            end_time = time.time()
            
            if result['success']:
                success_count += 1
                response_times.append(end_time - start_time)
            
            if (i + 1) % 10 == 0:
                logger.info(f"Progress: {i + 1}/{num_requests}")
        
        if response_times:
            avg_response_time = np.mean(response_times)
            min_response_time = np.min(response_times)
            max_response_time = np.max(response_times)
            p95_response_time = np.percentile(response_times, 95)
            
            throughput = success_count / sum(response_times)
            
            performance_metrics = {
                'total_requests': num_requests,
                'successful_requests': success_count,
                'success_rate': success_count / num_requests,
                'avg_response_time': avg_response_time,
                'min_response_time': min_response_time,
                'max_response_time': max_response_time,
                'p95_response_time': p95_response_time,
                'throughput': throughput
            }
            
            logger.info("Performance test completed", **performance_metrics)
            return performance_metrics
        else:
            logger.error("No successful requests in performance test")
            return {'error': 'No successful requests'}
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all API tests."""
        logger.info("Starting API tests")
        
        results = {
            'health': self.test_health_endpoint(),
            'model_info': self.test_model_info(),
            'prediction': self.test_prediction_endpoint(),
            'batch_prediction': self.test_batch_prediction(),
            'metrics': self.test_metrics_endpoint()
        }
        
        # Count successful tests
        successful_tests = sum(1 for result in results.values() if result.get('success', False))
        total_tests = len(results)
        
        logger.info("API tests completed", 
                   successful_tests=successful_tests,
                   total_tests=total_tests)
        
        return {
            'tests': results,
            'summary': {
                'successful_tests': successful_tests,
                'total_tests': total_tests,
                'success_rate': successful_tests / total_tests
            }
        }


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Test API endpoints")
    parser.add_argument("--base-url", default="http://localhost:8000",
                       help="Base URL of the API")
    parser.add_argument("--performance", action="store_true",
                       help="Run performance test")
    parser.add_argument("--num-requests", type=int, default=100,
                       help="Number of requests for performance test")
    parser.add_argument("--output", default=None,
                       help="Output file for test results")
    
    args = parser.parse_args()
    
    # Create API tester
    tester = APITester(args.base_url)
    
    # Run tests
    if args.performance:
        results = tester.performance_test(args.num_requests)
    else:
        results = tester.run_all_tests()
    
    # Print results
    if args.performance:
        print("\n📊 Performance Test Results:")
        print(f"   Total Requests: {results.get('total_requests', 0)}")
        print(f"   Successful Requests: {results.get('successful_requests', 0)}")
        print(f"   Success Rate: {results.get('success_rate', 0):.2%}")
        print(f"   Avg Response Time: {results.get('avg_response_time', 0):.3f}s")
        print(f"   P95 Response Time: {results.get('p95_response_time', 0):.3f}s")
        print(f"   Throughput: {results.get('throughput', 0):.2f} req/s")
    else:
        print("\n🧪 API Test Results:")
        for test_name, result in results['tests'].items():
            status = "✅ PASS" if result.get('success', False) else "❌ FAIL"
            print(f"   {test_name}: {status}")
        
        summary = results['summary']
        print(f"\n📈 Summary: {summary['successful_tests']}/{summary['total_tests']} tests passed")
    
    # Save results to file if specified
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Results saved to: {args.output}")


if __name__ == "__main__":
    main() 