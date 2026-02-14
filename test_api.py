"""
MobAI WMS API - Test Suite
Run with: pytest test_api.py -v
Or: python test_api.py
"""

import requests
import json
from datetime import datetime, timedelta

# API Base URL
BASE_URL = "http://localhost:8000"

# ANSI color codes for pretty output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    YELLOW = '\033[93m'
    END = '\033[0m'
    BOLD = '\033[1m'

def print_test(test_name):
    """Print test header"""
    print(f"\n{Colors.BLUE}{Colors.BOLD}{'='*60}{Colors.END}")
    print(f"{Colors.BLUE}{Colors.BOLD}TEST: {test_name}{Colors.END}")
    print(f"{Colors.BLUE}{Colors.BOLD}{'='*60}{Colors.END}")

def print_success(message):
    """Print success message"""
    print(f"{Colors.GREEN}✅ {message}{Colors.END}")

def print_error(message):
    """Print error message"""
    print(f"{Colors.RED}❌ {message}{Colors.END}")

def print_info(message):
    """Print info message"""
    print(f"{Colors.YELLOW}ℹ️  {message}{Colors.END}")

def print_json(data):
    """Pretty print JSON"""
    print(json.dumps(data, indent=2))

# ============================================================================
# TEST 1: Root Endpoint
# ============================================================================

def test_root():
    """Test root endpoint"""
    print_test("Root Endpoint")
    
    try:
        response = requests.get(f"{BASE_URL}/")
        
        assert response.status_code == 200, "Status code should be 200"
        data = response.json()
        
        assert "service" in data, "Response should contain 'service'"
        assert data["status"] == "running", "Status should be 'running'"
        
        print_success("Root endpoint working")
        print_info("Response:")
        print_json(data)
        
        return True
        
    except Exception as e:
        print_error(f"Root endpoint failed: {e}")
        return False

# ============================================================================
# TEST 2: Health Check
# ============================================================================

def test_health():
    """Test health check endpoint"""
    print_test("Health Check")
    
    try:
        response = requests.get(f"{BASE_URL}/health")
        
        assert response.status_code == 200, "Status code should be 200"
        data = response.json()
        
        assert data["status"] == "healthy", "Status should be healthy"
        assert "models" in data, "Should return models info"
        assert "data" in data, "Should return data info"
        
        print_success("Health check passed")
        print_info("Response:")
        print_json(data)
        
        return True
        
    except Exception as e:
        print_error(f"Health check failed: {e}")
        return False

# ============================================================================
# TEST 3: Forecasting - Single Product
# ============================================================================

def test_forecast_single():
    """Test forecasting for single product"""
    print_test("Forecasting - Single Product")
    
    try:
        tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
        
        payload = {
            "product_ids": [31554],
            "date": tomorrow
        }
        
        print_info(f"Request:")
        print_json(payload)
        
        response = requests.post(f"{BASE_URL}/predict", json=payload)
        
        assert response.status_code == 200, "Status code should be 200"
        data = response.json()
        
        assert len(data) == 1, "Should return 1 forecast"
        assert data[0]["product_id"] == 31554, "Product ID should match"
        assert "predicted_demand" in data[0], "Should have predicted_demand"
        assert "probability" in data[0], "Should have probability"
        
        print_success("Single product forecast working")
        print_info("Response:")
        print_json(data)
        
        return True
        
    except Exception as e:
        print_error(f"Single forecast failed: {e}")
        return False

# ============================================================================
# TEST 4: Forecasting - Multiple Products
# ============================================================================

def test_forecast_multiple():
    """Test forecasting for multiple products"""
    print_test("Forecasting - Multiple Products")
    
    try:
        tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
        
        payload = {
            "product_ids": [31554, 31565, 34015, 31557, 31993],
            "date": tomorrow
        }
        
        print_info(f"Request:")
        print_json(payload)
        
        response = requests.post(f"{BASE_URL}/predict", json=payload)
        
        assert response.status_code == 200, "Status code should be 200"
        data = response.json()
        
        assert len(data) == 5, "Should return 5 forecasts"
        
        # Check each forecast
        for forecast in data:
            assert "product_id" in forecast
            assert "predicted_demand" in forecast
            assert "probability" in forecast
            assert forecast["predicted_demand"] >= 0, "Demand should be non-negative"
            assert 0 <= forecast["probability"] <= 1, "Probability should be 0-1"
        
        print_success("Multiple product forecast working")
        print_info(f"Sample response (first 2):")
        print_json(data[:2])
        
        # Print summary
        total_demand = sum(f["predicted_demand"] for f in data)
        avg_prob = sum(f["probability"] for f in data) / len(data)
        
        print_info(f"\nSummary:")
        print(f"  Total forecasted demand: {total_demand:.2f}")
        print(f"  Average probability: {avg_prob:.4f}")
        
        return True
        
    except Exception as e:
        print_error(f"Multiple forecast failed: {e}")
        return False

# ============================================================================
# TEST 5: Storage Assignment
# ============================================================================

def test_storage_assignment():
    """Test storage location assignment"""
    print_test("Storage Assignment")
    
    try:
        # Test high-priority product
        payload = {
            "product_id": 31554,  # High-priority product
            "quantity": 100
        }
        
        print_info(f"Request (high-priority product):")
        print_json(payload)
        
        response = requests.post(f"{BASE_URL}/assign-storage", json=payload)
        
        assert response.status_code == 200, "Status code should be 200"
        data = response.json()
        
        assert data["product_id"] == 31554, "Product ID should match"
        assert "assigned_location" in data, "Should have assigned_location"
        assert "distance_to_expedition" in data, "Should have distance"
        assert "floor" in data, "Should have floor"
        assert "priority_score" in data, "Should have priority_score"
        
        print_success("Storage assignment working")
        print_info("Response:")
        print_json(data)
        
        # Test low-priority product
        payload2 = {
            "product_id": 31339,  # Low-priority product
            "quantity": 50
        }
        
        print_info(f"\nRequest (low-priority product):")
        print_json(payload2)
        
        response2 = requests.post(f"{BASE_URL}/assign-storage", json=payload2)
        data2 = response2.json()
        
        print_info("Response:")
        print_json(data2)
        
        print_info(f"\nComparison:")
        print(f"  High-priority distance: {data['distance_to_expedition']:.1f}m")
        print(f"  Low-priority distance: {data2['distance_to_expedition']:.1f}m")
        
        return True
        
    except Exception as e:
        print_error(f"Storage assignment failed: {e}")
        return False

# ============================================================================
# TEST 6: Picking Optimization
# ============================================================================

def test_picking_optimization():
    """Test picking route optimization (multi-chariot)"""
    print_test("Picking Route Optimization (Multi-Chariot)")
    
    try:
        payload = {
            "items": [
                {"product_id": 31554, "quantity": 10, "location_id": 398},
                {"product_id": 31565, "quantity": 20, "location_id": 404},
                {"product_id": 34015, "quantity": 15, "location_id": 407},
                {"product_id": 31557, "quantity": 5, "location_id": 1331},
                {"product_id": 31993, "quantity": 8, "location_id": 406}
            ],
            "chariot_capacity_kg": 300,
            "max_chariots": 3
        }
        
        print_info(f"Request ({len(payload['items'])} items):")
        print_json(payload)
        
        response = requests.post(f"{BASE_URL}/optimize-picking", json=payload)
        
        assert response.status_code == 200, "Status code should be 200"
        data = response.json()
        
        assert "chariots" in data, "Should have chariots"
        assert "total_distance" in data, "Should have total_distance"
        assert "total_items" in data, "Should have total_items"
        assert "total_chariots_used" in data, "Should have total_chariots_used"
        assert "efficiency_improvement" in data, "Should have efficiency"
        assert "congestion_warnings" in data, "Should have congestion_warnings"
        
        assert data["total_items"] == 5, "Should have 5 items"
        
        print_success("Picking optimization working")
        print_info("Response summary:")
        print(f"  Total distance: {data['total_distance']:.1f}m")
        print(f"  Total items: {data['total_items']}")
        print(f"  Chariots used: {data['total_chariots_used']}")
        print(f"  Efficiency improvement: {data['efficiency_improvement']:.1f}%")
        
        for chariot in data["chariots"]:
            print_info(f"\nChariot {chariot['chariot_id']} ({chariot['items_count']} items, {chariot['total_weight_kg']}kg):")
            for step in chariot["route"]:
                prod = step['product_id'] if step['product_id'] else 'RETURN'
                print(f"  Step {step['step']}: {step['location_code']} "
                      f"(Product: {prod}, Zone: {step['zone']}, Dist: {step['distance_from_previous']:.1f}m)")
        
        if data["congestion_warnings"]:
            print_info("\nCongestion warnings:")
            for w in data["congestion_warnings"]:
                print(f"  {w}")
        
        return True
        
    except Exception as e:
        print_error(f"Picking optimization failed: {e}")
        return False

# ============================================================================
# TEST 7: Simulation
# ============================================================================

def test_simulation():
    """Test simulation endpoint with ingoing/outgoing events"""
    print_test("Simulation")
    
    try:
        payload = {
            "events": [
                {"date": "2026-01-10", "product_id": 31554, "quantity": 50, "flow_type": "ingoing"},
                {"date": "2026-01-10", "product_id": 31565, "quantity": 30, "flow_type": "ingoing"},
                {"date": "2026-01-11", "product_id": 34015, "quantity": 20, "flow_type": "ingoing"},
                {"date": "2026-01-12", "product_id": 31554, "quantity": 10, "flow_type": "outgoing"},
                {"date": "2026-01-12", "product_id": 31565, "quantity": 5, "flow_type": "outgoing"},
                {"date": "2026-01-13", "product_id": 99999, "quantity": 5, "flow_type": "outgoing"}
            ],
            "reset_state": True
        }
        
        print_info(f"Request ({len(payload['events'])} events):")
        print_json(payload)
        
        response = requests.post(f"{BASE_URL}/simulate", json=payload)
        
        assert response.status_code == 200, f"Status code should be 200, got {response.status_code}"
        data = response.json()
        
        assert data["total_events_processed"] == 6, "Should process 6 events"
        assert "actions" in data, "Should have actions"
        assert "final_warehouse_state" in data, "Should have final warehouse state"
        assert "assumptions" in data, "Should have assumptions"
        
        # Check ingoing events succeeded
        ingoing_actions = [a for a in data["actions"] if a["flow_type"] == "ingoing"]
        assert all(a["success"] for a in ingoing_actions), "All ingoing should succeed"
        
        # Check outgoing for stored product succeeded
        outgoing_ok = [a for a in data["actions"] if a["flow_type"] == "outgoing" and a["product_id"] in [31554, 31565]]
        assert all(a["success"] for a in outgoing_ok), "Outgoing for stored products should succeed"
        
        # Check outgoing for unknown product failed
        outgoing_fail = [a for a in data["actions"] if a["product_id"] == 99999]
        assert len(outgoing_fail) == 1 and not outgoing_fail[0]["success"], "Outgoing for unstored product should fail"
        
        print_success("Simulation working")
        print_info("Actions:")
        for a in data["actions"]:
            status = "OK" if a["success"] else "FAIL"
            print(f"  [{status}] {a['date']} | {a['flow_type']} | Product {a['product_id']} x{a['quantity']} | {a['action'][:60]}")
        
        print_info(f"\nFinal warehouse state:")
        print_json(data["final_warehouse_state"])
        
        if data["assumptions"]:
            print_info(f"\nAssumptions ({len(data['assumptions'])}):")
            for a in data["assumptions"][:3]:
                print(f"  - {a}")
        
        return True
        
    except Exception as e:
        print_error(f"Simulation failed: {e}")
        return False

# ============================================================================
# TEST 8: Warehouse State
# ============================================================================

def test_warehouse_state():
    """Test warehouse state endpoint"""
    print_test("Warehouse State")
    
    try:
        # Reset first
        requests.post(f"{BASE_URL}/reset-warehouse")
        
        # Store a product
        requests.post(f"{BASE_URL}/assign-storage", json={"product_id": 31554, "quantity": 50})
        
        response = requests.get(f"{BASE_URL}/warehouse-state")
        assert response.status_code == 200
        data = response.json()
        
        assert "occupancy" in data, "Should have occupancy"
        assert data["occupancy"]["occupied_slots"] >= 1, "Should have at least 1 occupied slot"
        assert data["product_count"] >= 1, "Should have at least 1 product"
        assert len(data["occupied_slots_detail"]) >= 1, "Should have slot details"
        
        print_success("Warehouse state working")
        print_info("Occupancy:")
        print_json(data["occupancy"])
        print_info(f"Products stored: {data['product_count']}")
        
        # Reset
        reset_resp = requests.post(f"{BASE_URL}/reset-warehouse")
        assert reset_resp.status_code == 200
        print_success("Warehouse reset working")
        
        return True
        
    except Exception as e:
        print_error(f"Warehouse state failed: {e}")
        return False

# ============================================================================
# TEST 9: Explainability (XAI)
# ============================================================================

def test_explainability():
    """Test XAI/explainability endpoint"""
    print_test("Explainability (XAI)")
    
    try:
        payload = {
            "product_id": 31554,
            "date": "2026-01-15"
        }
        
        response = requests.post(f"{BASE_URL}/explain", json=payload)
        assert response.status_code == 200
        data = response.json()
        
        assert "segment" in data, "Should have segment"
        assert "prophet_baseline" in data, "Should have prophet_baseline"
        assert "classifier_probability" in data, "Should have classifier_probability"
        assert "final_prediction" in data, "Should have final_prediction"
        assert "confidence_interval" in data, "Should have confidence_interval"
        assert "top_factors" in data, "Should have top_factors"
        assert "model_components" in data, "Should have model_components"
        assert "assumptions" in data, "Should have assumptions"
        assert len(data["top_factors"]) > 0, "Should have at least 1 factor"
        
        print_success("Explainability working")
        print_info(f"Product {data['product_id']} (Segment: {data['segment']}):")
        print(f"  Prophet baseline: {data['prophet_baseline']}")
        print(f"  Classifier probability: {data['classifier_probability']}")
        print(f"  Final prediction: {data['final_prediction']}")
        print(f"  Confidence: [{data['confidence_interval']['low']}, {data['confidence_interval']['high']}]")
        print_info("\nTop factors:")
        for f in data["top_factors"][:4]:
            print(f"  [{f['impact']}] {f['factor']}: {f['value']}")
        
        return True
        
    except Exception as e:
        print_error(f"Explainability failed: {e}")
        return False

# ============================================================================
# TEST 10: Model Info
# ============================================================================

def test_model_info():
    """Test model info endpoint"""
    print_test("Model Info")
    
    try:
        response = requests.get(f"{BASE_URL}/model-info")
        assert response.status_code == 200
        data = response.json()
        
        assert "model_version" in data, "Should have model_version"
        assert "performance" in data, "Should have performance"
        assert "architecture" in data, "Should have architecture"
        assert "assumptions" in data, "Should have assumptions"
        assert "limitations" in data, "Should have limitations"
        
        print_success("Model info working")
        print_info(f"Model: {data['model_version']}")
        print_info(f"Performance: {json.dumps(data['performance'], indent=2)}")
        print_info(f"Assumptions: {len(data['assumptions'])} documented")
        
        return True
        
    except Exception as e:
        print_error(f"Model info failed: {e}")
        return False

# ============================================================================
# TEST 11: Error Handling
# ============================================================================

def test_error_handling():
    """Test API error handling"""
    print_test("Error Handling")
    
    tests_passed = 0
    
    # Test 1: Invalid date format
    try:
        payload = {"product_ids": [31554], "date": "invalid-date"}
        response = requests.post(f"{BASE_URL}/predict", json=payload)
        
        assert response.status_code in [422, 500], "Should return error for invalid date"
        print_success("Invalid date format handled correctly")
        tests_passed += 1
    except Exception as e:
        print_error(f"Invalid date test failed: {e}")
    
    # Test 2: Empty picking list
    try:
        payload = {"items": []}
        response = requests.post(f"{BASE_URL}/optimize-picking", json=payload)
        
        assert response.status_code == 400, "Should return 400 for empty list"
        print_success("Empty picking list handled correctly")
        tests_passed += 1
    except Exception as e:
        print_error(f"Empty list test failed: {e}")
    
    # Test 3: Invalid location ID
    try:
        payload = {
            "items": [
                {"product_id": 31554, "quantity": 10, "location_id": 999999}
            ]
        }
        response = requests.post(f"{BASE_URL}/optimize-picking", json=payload)
        
        assert response.status_code == 404, "Should return 404 for invalid location"
        print_success("Invalid location handled correctly")
        tests_passed += 1
    except Exception as e:
        print_error(f"Invalid location test failed: {e}")
    
    print_info(f"\nError handling tests passed: {tests_passed}/3")
    
    return tests_passed == 3

# ============================================================================
# RUN ALL TESTS
# ============================================================================

def run_all_tests():
    """Run all tests"""
    print(f"\n{Colors.BOLD}{'='*60}{Colors.END}")
    print(f"\n{Colors.BOLD}MobAI WMS API v2.0 - COMPREHENSIVE TEST SUITE{Colors.END}")
    print(f"{Colors.BOLD}{'='*60}{Colors.END}")
    print(f"\n{Colors.YELLOW}Testing API at: {BASE_URL}{Colors.END}\n")
    
    results = {
        "Root Endpoint": test_root(),
        "Health Check": test_health(),
        "Forecast - Single": test_forecast_single(),
        "Forecast - Multiple": test_forecast_multiple(),
        "Storage Assignment": test_storage_assignment(),
        "Picking Optimization": test_picking_optimization(),
        "Simulation": test_simulation(),
        "Warehouse State": test_warehouse_state(),
        "Explainability": test_explainability(),
        "Model Info": test_model_info(),
        "Error Handling": test_error_handling()
    }
    
    # Summary
    print(f"\n{Colors.BOLD}{'='*60}{Colors.END}")
    print(f"{Colors.BOLD}TEST SUMMARY{Colors.END}")
    print(f"{Colors.BOLD}{'='*60}{Colors.END}\n")
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = f"{Colors.GREEN}✅ PASS{Colors.END}" if result else f"{Colors.RED}❌ FAIL{Colors.END}"
        print(f"{test_name:<30} {status}")
    
    print(f"\n{Colors.BOLD}Total: {passed}/{total} tests passed{Colors.END}")
    
    if passed == total:
        print(f"{Colors.GREEN}{Colors.BOLD}\n🎉 ALL TESTS PASSED! 🎉{Colors.END}\n")
    else:
        print(f"{Colors.RED}{Colors.BOLD}\n⚠️  Some tests failed{Colors.END}\n")
    
    return passed == total

if __name__ == "__main__":
    try:
        success = run_all_tests()
        exit(0 if success else 1)
    except KeyboardInterrupt:
        print(f"\n\n{Colors.YELLOW}Tests interrupted by user{Colors.END}\n")
        exit(1)
    except Exception as e:
        print(f"\n{Colors.RED}Test suite failed: {e}{Colors.END}\n")
        exit(1)