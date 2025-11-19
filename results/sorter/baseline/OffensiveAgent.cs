using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;

public class OffensiveAgent: Agent
{
    public Transform ball;
    public Rigidbody ballRb;
    public Transform teammate;
    public Transform opponentGoal; 
    public Transform opponentAgent;
    public float fieldLength = 40f;

    private int stepsSinceLastTouch = 0;
    private float lastAction = 0;   //impl soon

    private Rigidbody agentRb;
    private Rigidbody opponentRb;

    private const float MOVEMENT_MULTIPLIER = 10f; 

    public override void Initialize()
    {
        agentRb = GetComponent<Rigidbody>();
        if (opponentAgent != null)
        {
            opponentRb = opponentAgent.GetComponent<Rigidbody>();
        }
        if (ballRb == null && ball != null)
        {
            ballRb = ball.GetComponent<Rigidbody>();
        }
    }

    public override void OnEpisodeBegin()
    {
        stepsSinceLastTouch = 0;
        lastAction = 0;
    
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        //calc of 
        float distanceToBall = Vector3.Distance(transform.position, ball.position);
        float fieldHalf = fieldLength / 2f;

        //ball distance
        sensor.AddObservation(distanceToBall); 

        //opponent position 
        if (opponentAgent != null && opponentRb != null)
        {
            // nearest opponent distance (records only one so adequate for both games)
            sensor.AddObservation(Vector3.Distance(transform.position, opponentAgent.position));

            // opponent relative velocity (3 floats)
            sensor.AddObservation(transform.InverseTransformDirection(opponentRb.velocity));
        }
        else
        {
            sensor.AddObservation(0f);
            sensor.AddObservation(Vector3.zero); 
        }

        //time since last touch 
        sensor.AddObservation(stepsSinceLastTouch);

    }

    public override void OnActionReceived(ActionBuffers actions){
        stepsSinceLastTouch++;


        //tbd
    }
        


    private void OnCollisionEnter(Collision collision)
    {
        if (collision.gameObject.transform == ball)
        {
            stepsSinceLastTouch = 0;
        }
    }



}