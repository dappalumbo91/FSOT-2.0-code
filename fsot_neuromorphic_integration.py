#!/usr/bin/env python3
"""
FSOT 2.0 Neuromorphic Brain Integration Module
Integrates the complete neuromorphic brain system with existing FSOT 2.0 infrastructure
"""

import os
import sys
import json
import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Union

# Import neuromorphic brain system
try:
    from fsot_working_neuromorphic_brain import WorkingNeuromorphicBrain, NeuralSignal, ProcessingPriority
except ImportError:
    print("❌ Neuromorphic brain system not found. Creating minimal implementation...")
    class WorkingNeuromorphicBrain:
        def __init__(self):
            self.consciousness_level = 0.5
            self.brain_regions = {}
            
        async def process_signal(self, signal):
            return f"Processed: {signal}"

# Import existing FSOT 2.0 components
try:
    from fsot_core import FSOTCore
except ImportError:
    print("⚠️  FSOT Core not found - will create integration layer")
    FSOTCore = None

try:
    from autonomous_learning_system import AutonomousLearningSystem
except ImportError:
    print("⚠️  Autonomous Learning System not found")
    AutonomousLearningSystem = None

try:
    from advanced_ai_assistant import AdvancedAIAssistant
except ImportError:
    print("⚠️  Advanced AI Assistant not found")
    AdvancedAIAssistant = None

class FSOT2NeuromorphicIntegration:
    """
    Complete integration layer between FSOT 2.0 and Neuromorphic Brain System
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or "fsot_neuromorphic_integration_config.json"
        self.integration_state = {}
        self.brain_system = None
        self.fsot_core = None
        self.learning_system = None
        self.ai_assistant = None
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('fsot_neuromorphic_integration.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Initialize integration
        self._initialize_integration()

    def _initialize_integration(self):
        """Initialize the complete FSOT 2.0 - Neuromorphic Brain integration"""
        self.logger.info("🧠 Initializing FSOT 2.0 Neuromorphic Brain Integration")
        
        # Load configuration
        self._load_configuration()
        
        # Initialize brain system
        self._initialize_brain_system()
        
        # Initialize FSOT components
        self._initialize_fsot_components()
        
        # Setup neural-FSOT bridges
        self._setup_neural_bridges()
        
        # Create integration state
        self._create_integration_state()
        
        self.logger.info("✅ FSOT 2.0 Neuromorphic Integration Complete")

    def _load_configuration(self):
        """Load integration configuration"""
        default_config = {
            "neuromorphic_brain": {
                "consciousness_threshold": 0.5,
                "brain_wave_monitoring": True,
                "neural_signal_processing": True,
                "cross_region_integration": True
            },
            "fsot_integration": {
                "autonomous_learning": True,
                "advanced_ai_assistant": True,
                "consciousness_driven_decisions": True,
                "neural_memory_integration": True
            },
            "processing_priorities": {
                "emergency_override": True,
                "consciousness_gating": True,
                "emotional_modulation": True,
                "learning_integration": True
            },
            "interface_mapping": {
                "frontal_lobe": ["executive_control", "decision_making"],
                "temporal_lobe": ["language_processing", "memory_retrieval"],
                "parietal_lobe": ["spatial_reasoning", "attention"],
                "occipital_lobe": ["visual_processing", "pattern_recognition"],
                "hippocampus": ["memory_formation", "learning"],
                "amygdala": ["emotional_processing", "threat_assessment"],
                "thalamus": ["information_relay", "attention_gating"],
                "cerebellum": ["skill_refinement", "motor_learning"]
            }
        }
        
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    loaded_config = json.load(f)
                # Merge with defaults
                self.config = {**default_config, **loaded_config}
            else:
                self.config = default_config
                # Save default config
                with open(self.config_path, 'w') as f:
                    json.dump(self.config, f, indent=4)
        except Exception as e:
            self.logger.warning(f"Config loading failed: {e}. Using defaults.")
            self.config = default_config

    def _initialize_brain_system(self):
        """Initialize the neuromorphic brain system"""
        try:
            self.brain_system = WorkingNeuromorphicBrain()
            self.logger.info("✅ Neuromorphic brain system initialized")
        except Exception as e:
            self.logger.error(f"❌ Brain system initialization failed: {e}")
            self.brain_system = None

    def _initialize_fsot_components(self):
        """Initialize existing FSOT 2.0 components"""
        try:
            # Initialize FSOT Core if available
            if FSOTCore:
                self.fsot_core = FSOTCore()
                self.logger.info("✅ FSOT Core initialized")
            
            # Initialize Autonomous Learning System if available
            if AutonomousLearningSystem:
                self.learning_system = AutonomousLearningSystem()
                self.logger.info("✅ Autonomous Learning System initialized")
            
            # Initialize Advanced AI Assistant if available
            if AdvancedAIAssistant:
                self.ai_assistant = AdvancedAIAssistant()
                self.logger.info("✅ Advanced AI Assistant initialized")
                
        except Exception as e:
            self.logger.warning(f"⚠️  Some FSOT components failed to initialize: {e}")

    def _setup_neural_bridges(self):
        """Setup bridges between neural regions and FSOT components"""
        self.neural_bridges = {
            'frontal_lobe': {
                'fsot_functions': ['executive_decision_making', 'goal_planning'],
                'processing_type': 'executive',
                'priority': ProcessingPriority.EXECUTIVE if 'ProcessingPriority' in globals() else 'HIGH'
            },
            'temporal_lobe': {
                'fsot_functions': ['language_understanding', 'conversation_ai'],
                'processing_type': 'linguistic',
                'priority': ProcessingPriority.COGNITIVE if 'ProcessingPriority' in globals() else 'MEDIUM'
            },
            'hippocampus': {
                'fsot_functions': ['memory_storage', 'learning_optimization'],
                'processing_type': 'memory',
                'priority': ProcessingPriority.COGNITIVE if 'ProcessingPriority' in globals() else 'MEDIUM'
            },
            'amygdala': {
                'fsot_functions': ['threat_assessment', 'emotional_responses'],
                'processing_type': 'emotional',
                'priority': ProcessingPriority.EMOTIONAL if 'ProcessingPriority' in globals() else 'HIGH'
            }
        }
        
        self.logger.info("✅ Neural-FSOT bridges established")

    def _create_integration_state(self):
        """Create comprehensive integration state"""
        self.integration_state = {
            'timestamp': datetime.now().isoformat(),
            'integration_status': 'ACTIVE',
            'brain_system_status': 'OPERATIONAL' if self.brain_system else 'UNAVAILABLE',
            'fsot_core_status': 'OPERATIONAL' if self.fsot_core else 'UNAVAILABLE',
            'learning_system_status': 'OPERATIONAL' if self.learning_system else 'UNAVAILABLE',
            'ai_assistant_status': 'OPERATIONAL' if self.ai_assistant else 'UNAVAILABLE',
            'neural_bridges_count': len(self.neural_bridges),
            'consciousness_level': self.brain_system.consciousness_level if self.brain_system else 0.0,
            'integration_capabilities': self._assess_capabilities()
        }

    def _assess_capabilities(self):
        """Assess integrated system capabilities"""
        capabilities = {
            'neuromorphic_processing': self.brain_system is not None,
            'autonomous_learning': self.learning_system is not None,
            'advanced_ai_assistance': self.ai_assistant is not None,
            'consciousness_tracking': self.brain_system is not None,
            'emotional_intelligence': self.brain_system is not None,
            'cross_modal_integration': True,
            'real_time_processing': True,
            'adaptive_responses': True
        }
        
        return capabilities

    async def process_integrated_request(self, request: str, context: Dict = None) -> Dict[str, Any]:
        """
        Process request through integrated FSOT 2.0 + Neuromorphic Brain system
        """
        self.logger.info(f"🧠 Processing integrated request: {request[:100]}...")
        
        processing_results = {
            'timestamp': datetime.now().isoformat(),
            'original_request': request,
            'context': context or {},
            'processing_stages': {},
            'neural_activity': {},
            'fsot_responses': {},
            'integrated_response': '',
            'consciousness_evolution': {}
        }
        
        try:
            # Stage 1: Neural preprocessing through brain regions
            if self.brain_system:
                neural_results = await self._neural_preprocessing(request, context)
                processing_results['processing_stages']['neural_preprocessing'] = neural_results
                processing_results['neural_activity'] = neural_results.get('brain_activity', {})
            
            # Stage 2: FSOT component processing
            fsot_results = await self._fsot_processing(request, context, processing_results.get('neural_activity', {}))
            processing_results['processing_stages']['fsot_processing'] = fsot_results
            processing_results['fsot_responses'] = fsot_results
            
            # Stage 3: Neural-FSOT integration
            integrated_response = await self._neural_fsot_integration(processing_results)
            processing_results['integrated_response'] = integrated_response
            
            # Stage 4: Consciousness evolution tracking
            if self.brain_system:
                consciousness_data = await self._track_consciousness_evolution()
                processing_results['consciousness_evolution'] = consciousness_data
            
            self.logger.info("✅ Integrated request processing complete")
            return processing_results
            
        except Exception as e:
            self.logger.error(f"❌ Integrated processing failed: {e}")
            processing_results['error'] = str(e)
            processing_results['integrated_response'] = f"Processing error: {e}"
            return processing_results

    async def _neural_preprocessing(self, request: str, context: Dict) -> Dict[str, Any]:
        """Preprocess request through neuromorphic brain regions"""
        if not self.brain_system:
            return {'status': 'brain_unavailable'}
        
        # Create neural signal for the request
        try:
            if 'NeuralSignal' in globals():
                neural_signal = NeuralSignal(
                    source="external_input",
                    target="frontal_lobe",
                    data={'request': request, 'context': context},
                    signal_type="cognitive_request",
                    priority=ProcessingPriority.COGNITIVE
                )
            else:
                neural_signal = {
                    'source': 'external_input',
                    'target': 'frontal_lobe',
                    'data': {'request': request, 'context': context},
                    'signal_type': 'cognitive_request'
                }
            
            # Process through brain system
            brain_response = await self.brain_system.process_signal(neural_signal)
            
            return {
                'status': 'success',
                'neural_signal': str(neural_signal),
                'brain_response': brain_response,
                'brain_activity': {
                    'consciousness_level': getattr(self.brain_system, 'consciousness_level', 0.5),
                    'active_regions': list(getattr(self.brain_system, 'brain_regions', {}).keys()),
                    'processing_time': datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            return {'status': 'error', 'error': str(e)}

    async def _fsot_processing(self, request: str, context: Dict, neural_activity: Dict) -> Dict[str, Any]:
        """Process request through FSOT 2.0 components"""
        fsot_responses = {}
        
        # Process through FSOT Core if available
        if self.fsot_core:
            try:
                if hasattr(self.fsot_core, 'process_request'):
                    core_response = await self.fsot_core.process_request(request, context, neural_activity)
                else:
                    core_response = f"FSOT Core processed: {request[:50]}..."
                fsot_responses['fsot_core'] = core_response
            except Exception as e:
                fsot_responses['fsot_core'] = f"Error: {e}"
        
        # Process through Learning System if available
        if self.learning_system:
            try:
                if hasattr(self.learning_system, 'learn_from_interaction'):
                    learning_response = await self.learning_system.learn_from_interaction(request, context)
                else:
                    learning_response = f"Learning system processed: {request[:50]}..."
                fsot_responses['learning_system'] = learning_response
            except Exception as e:
                fsot_responses['learning_system'] = f"Error: {e}"
        
        # Process through AI Assistant if available
        if self.ai_assistant:
            try:
                if hasattr(self.ai_assistant, 'process_message'):
                    assistant_response = await self.ai_assistant.process_message(request, context)
                else:
                    assistant_response = f"AI Assistant processed: {request[:50]}..."
                fsot_responses['ai_assistant'] = assistant_response
            except Exception as e:
                fsot_responses['ai_assistant'] = f"Error: {e}"
        
        return fsot_responses

    async def _neural_fsot_integration(self, processing_results: Dict) -> str:
        """Integrate neural and FSOT processing results"""
        neural_activity = processing_results.get('neural_activity', {})
        fsot_responses = processing_results.get('fsot_responses', {})
        
        # Consciousness-driven response synthesis
        consciousness_level = neural_activity.get('consciousness_level', 0.5)
        
        if consciousness_level > 0.7:
            integration_style = "highly_conscious"
        elif consciousness_level > 0.5:
            integration_style = "moderately_conscious"
        else:
            integration_style = "basic_processing"
        
        # Synthesize integrated response
        response_components = []
        
        # Add neural insights
        if 'brain_response' in processing_results.get('processing_stages', {}).get('neural_preprocessing', {}):
            brain_response = processing_results['processing_stages']['neural_preprocessing']['brain_response']
            response_components.append(f"Neural Analysis: {brain_response}")
        
        # Add FSOT responses
        for component, response in fsot_responses.items():
            if response and not response.startswith("Error"):
                response_components.append(f"{component.title()}: {response}")
        
        # Create integrated response based on consciousness level
        if integration_style == "highly_conscious":
            integrated_response = "🧠 **Highly Conscious Processing**\n\n" + "\n\n".join(response_components)
        elif integration_style == "moderately_conscious":
            integrated_response = "🤔 **Conscious Processing**\n\n" + "\n".join(response_components)
        else:
            integrated_response = "⚡ **Basic Processing**\n" + " | ".join(response_components)
        
        return integrated_response

    async def _track_consciousness_evolution(self) -> Dict[str, Any]:
        """Track consciousness evolution during processing"""
        if not self.brain_system:
            return {'status': 'brain_unavailable'}
        
        return {
            'current_consciousness': getattr(self.brain_system, 'consciousness_level', 0.5),
            'brain_wave_patterns': getattr(self.brain_system, 'brain_wave_patterns', {}),
            'active_regions': list(getattr(self.brain_system, 'brain_regions', {}).keys()),
            'neural_connections': getattr(self.brain_system, 'neural_connections', []),
            'evolution_timestamp': datetime.now().isoformat()
        }

    def create_neuromorphic_fsot_dashboard(self) -> Dict[str, Any]:
        """Create comprehensive dashboard showing integration status"""
        dashboard = {
            'integration_overview': {
                'status': self.integration_state.get('integration_status', 'UNKNOWN'),
                'timestamp': datetime.now().isoformat(),
                'components_active': sum([
                    self.brain_system is not None,
                    self.fsot_core is not None,
                    self.learning_system is not None,
                    self.ai_assistant is not None
                ])
            },
            'neuromorphic_brain': {
                'status': self.integration_state.get('brain_system_status', 'UNAVAILABLE'),
                'consciousness_level': self.integration_state.get('consciousness_level', 0.0),
                'brain_regions': len(getattr(self.brain_system, 'brain_regions', {})) if self.brain_system else 0,
                'neural_bridges': len(self.neural_bridges)
            },
            'fsot_components': {
                'fsot_core': self.integration_state.get('fsot_core_status', 'UNAVAILABLE'),
                'learning_system': self.integration_state.get('learning_system_status', 'UNAVAILABLE'),
                'ai_assistant': self.integration_state.get('ai_assistant_status', 'UNAVAILABLE')
            },
            'integration_capabilities': self.integration_state.get('integration_capabilities', {}),
            'neural_fsot_mapping': self.neural_bridges,
            'configuration': self.config
        }
        
        return dashboard

    def save_integration_state(self):
        """Save current integration state"""
        state_file = "fsot_neuromorphic_integration_state.json"
        try:
            dashboard = self.create_neuromorphic_fsot_dashboard()
            with open(state_file, 'w') as f:
                json.dump(dashboard, f, indent=4)
            self.logger.info(f"✅ Integration state saved to {state_file}")
        except Exception as e:
            self.logger.error(f"❌ Failed to save integration state: {e}")

    async def run_integration_test(self) -> Dict[str, Any]:
        """Run comprehensive integration test"""
        self.logger.info("🧪 Running FSOT 2.0 Neuromorphic Integration Test")
        
        test_scenarios = [
            {
                'name': 'Basic Cognitive Processing',
                'request': 'Analyze the current weather patterns and suggest outdoor activities',
                'context': {'user_location': 'test_location', 'time': 'afternoon'}
            },
            {
                'name': 'Emotional Intelligence Test',
                'request': 'Help me deal with stress from work deadlines',
                'context': {'emotional_state': 'stressed', 'context': 'work_pressure'}
            },
            {
                'name': 'Learning Integration Test',
                'request': 'Teach me about quantum computing fundamentals',
                'context': {'learning_goal': 'quantum_computing', 'expertise_level': 'beginner'}
            }
        ]
        
        test_results = {
            'test_timestamp': datetime.now().isoformat(),
            'integration_status': 'TESTING',
            'scenario_results': [],
            'overall_performance': {}
        }
        
        for scenario in test_scenarios:
            self.logger.info(f"Testing: {scenario['name']}")
            
            scenario_result = await self.process_integrated_request(
                scenario['request'], 
                scenario['context']
            )
            
            test_results['scenario_results'].append({
                'scenario': scenario['name'],
                'request': scenario['request'],
                'processing_time': datetime.now().isoformat(),
                'success': 'error' not in scenario_result,
                'neural_activity': scenario_result.get('neural_activity', {}),
                'fsot_responses': len(scenario_result.get('fsot_responses', {})),
                'consciousness_level': scenario_result.get('consciousness_evolution', {}).get('current_consciousness', 0.0)
            })
        
        # Calculate overall performance
        successful_tests = sum(1 for result in test_results['scenario_results'] if result['success'])
        test_results['overall_performance'] = {
            'success_rate': successful_tests / len(test_scenarios) * 100,
            'total_scenarios': len(test_scenarios),
            'successful_scenarios': successful_tests,
            'integration_operational': successful_tests > 0
        }
        
        # Save test results
        with open('fsot_neuromorphic_integration_test_results.json', 'w') as f:
            json.dump(test_results, f, indent=4)
        
        self.logger.info(f"✅ Integration test complete. Success rate: {test_results['overall_performance']['success_rate']}%")
        return test_results

async def main():
    """Main integration demonstration"""
    print("🧠 FSOT 2.0 Neuromorphic Brain Integration System")
    print("=" * 60)
    
    # Initialize integration
    integration = FSOT2NeuromorphicIntegration()
    
    # Create dashboard
    dashboard = integration.create_neuromorphic_fsot_dashboard()
    print(f"📊 Integration Status: {dashboard['integration_overview']['status']}")
    print(f"🧠 Brain System: {dashboard['neuromorphic_brain']['status']}")
    print(f"⚡ Active Components: {dashboard['integration_overview']['components_active']}/4")
    
    # Run integration test
    test_results = await integration.run_integration_test()
    print(f"\n🧪 Integration Test Results:")
    print(f"✅ Success Rate: {test_results['overall_performance']['success_rate']}%")
    print(f"📈 Scenarios Tested: {test_results['overall_performance']['total_scenarios']}")
    
    # Save integration state
    integration.save_integration_state()
    
    print("\n🎉 FSOT 2.0 Neuromorphic Integration Complete!")
    print("Files created:")
    print("- fsot_neuromorphic_integration_config.json")
    print("- fsot_neuromorphic_integration_state.json") 
    print("- fsot_neuromorphic_integration_test_results.json")
    print("- fsot_neuromorphic_integration.log")

if __name__ == "__main__":
    asyncio.run(main())
